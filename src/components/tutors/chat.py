# SPDX-FileCopyrightText: 2023 Mark Liffiton <liffiton@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
from collections import Counter
from collections.abc import AsyncGenerator, Iterator
from datetime import UTC, datetime

import msgspec
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    make_response,
    redirect,
    render_template,
    request,
    stream_with_context,
    url_for,
)
from werkzeug.wrappers.response import Response

from components.code_contexts import (
    ContextConfig,
    contexts_config_table,
)
from gened.access import (
    Access,
    RequireComponent,
    check_access,
    class_enabled_required,
    route_requires,
)
from gened.app_data import DataAccessError
from gened.auth import get_auth
from gened.classes import switch_class
from gened.db import get_db
from gened.llm import LLM, ChatMessage, with_llm

from . import prompts
from .chat_helpers import create_guided_chat, create_inquiry_chat, get_or_create_assessment_quizzes
from .data import chats_data_source, guided_tutor_config_table
from .data_types import ChatData, GuidedAnalysis, QuizAnswer, QuizKind, Usage, WarmupQuiz

MAX_MESSAGE_LEN = 10_000
TERMINAL_OBJECTIVE_STATUSES = frozenset({'completed', 'moved on'})

bp = Blueprint('tutors', __name__, url_prefix='/tutor', template_folder='templates')

# NOTE: Blueprint default access controls set in __init__ via availability_requirements


def guided_chat_complete(chat: ChatData) -> bool:
    """Return whether every objective in a guided chat is in a terminal state."""
    return (
        chat.mode == 'guided'
        and chat.analysis is not None
        and bool(chat.analysis.progress)
        and all(
            objective.status in TERMINAL_OBJECTIVE_STATUSES
            for objective in chat.analysis.progress
        )
    )


def _chat_blocked_message(chat: ChatData) -> str | None:
    if chat.warmup_quiz is not None and not chat.warmup_quiz.reviewed:
        return "Complete the warm-up quiz before starting the chat."
    if guided_chat_complete(chat):
        return "This tutor session is complete."
    return None


@bp.route("/new")
@bp.route("/new/<int:class_id>")
def new_chat_form(class_id: int | None = None) -> Response | str:
    auth = get_auth()

    if class_id is not None:
        success = switch_class(class_id)
        if not success:
            current_app.logger.warning(f"User {auth.user_id} failed to switch to class {class_id}.")
            # Can't access the specified class
            flash("Cannot access specified class.  Make sure you are logged in correctly before using this link.", "danger")
            return make_response(render_template("error.html"), 400)

        contexts = None
        tutors = None

        if 'ctx_name' in request.args:
            ctx_name = request.args['ctx_name']
            context = contexts_config_table.get_item_by_name(ctx_name)
            if not context:
                flash(f"Context not found: '{ctx_name}'", "danger")
                return make_response(render_template("error.html"), 400)
            contexts = {context.name: context.desc_html()}
        elif 'tutor_name' in request.args:
            tutor_name = request.args['tutor_name']
            tutor = guided_tutor_config_table.get_item_by_name(tutor_name)
            if not tutor:
                flash(f"Tutor not found: '{tutor_name}'", "danger")
                return make_response(render_template("error.html"), 400)
            tutors = [tutor]
        else:
            return make_response(render_template("error.html"), 400)

    else:
        # All contexts and all guided tutors
        contexts_list = contexts_config_table.get_items(available_only=True)
        # turn into format we can pass to js via JSON
        contexts = {ctx.name: ctx.desc_html() for ctx in contexts_list}

        # Get all pre-defined guided tutors that are available:
        #   current date anywhere on earth (using UTC+12) is at or after the saved date
        tutors = guided_tutor_config_table.get_items(available_only=True)

    # check if each feature is enabled
    if not check_access(RequireComponent("tutors", "inquiry")):
        contexts = None  # don't display the form
    if not check_access(RequireComponent("tutors", "guided")):
        tutors = None  # don't display the form

    recent_chats = chats_data_source.get_user_data(limit=10)

    return render_template("tutor_new_form.html", contexts=contexts, tutors=tutors, recent_chats=recent_chats)


@bp.route("/new/inquiry", methods=["POST"])
@route_requires(Access.CLASS_ENABLED, RequireComponent("tutors", feature="inquiry"))
@with_llm(spend_token=True)
def new_inquiry_chat(llm: LLM) -> Response:
    topic = request.form['topic']
    context: ContextConfig | None = None

    if context_name := request.form.get('context'):
        context = contexts_config_table.get_item_by_name(context_name)
        if context is None:
            flash(f"Context not found: {context_name}", "danger")
            return make_response(render_template("error.html"), 400)

    chat = create_inquiry_chat(topic, context)

    try:
        run_chat_round(llm, chat)
    except RuntimeError as e:
        current_app.logger.error(f"Error running inquiry chat round: {e}")
        # On error, erase the nascent chat and tell the user what happened.
        erase_chat(chat)
        flash(str(e), 'danger')
        return make_response(render_template("error.html"), 502)

    return redirect(url_for("tutors.chat_interface", chat_id=chat.id))


@bp.route("/new/guided", methods=["POST"])
@route_requires(Access.CLASS_ENABLED, RequireComponent("tutors", feature="guided"))
@with_llm(spend_token=True)
def new_guided_chat(llm: LLM) -> Response:
    auth = get_auth()
    assert auth.cur_class is not None

    tutor_id = request.form['tutor_id']
    tutor_config = guided_tutor_config_table.get_item_by_id(int(tutor_id))

    if tutor_config is None:
        flash("Tutor not found.", "danger")
        return make_response(render_template("error.html"), 400)

    warmup_quiz = None
    wrapup_quiz = None
    if tutor_config.objectives:
        try:
            warmup_quiz, wrapup_quiz = asyncio.run(get_or_create_assessment_quizzes(tutor_config, llm))
        except (msgspec.DecodeError, msgspec.ValidationError, ValueError) as e:
            current_app.logger.error(f"Failed to prepare assessment quizzes for tutor {tutor_config.row_id}: {e}")
            flash("The warm-up and wrap-up quizzes could not be prepared. Please retry or ask your instructor for help.", "danger")
            return make_response(render_template("error.html"), 502)

    chat = create_guided_chat(
        tutor_config,
        warmup_quiz=warmup_quiz,
        wrapup_quiz=wrapup_quiz,
    )

    if tutor_config.opening_message:
        # Use the pre-generated/instructor-written opening message; no LLM call needed.
        chat.messages.append({
            'role': 'assistant',
            'content': tutor_config.opening_message,
        })
        save_chat(chat)
    else:
        # Fallback: no saved opening message, generate one live.
        try:
            run_chat_round(llm, chat)
        except RuntimeError as e:
            current_app.logger.error(f"Error running guided chat round: {e}")
            # On error, erase the nascent chat and tell the user what happened.
            erase_chat(chat)
            flash(str(e), 'danger')
            return make_response(render_template("error.html"), 502)

    return redirect(url_for("tutors.chat_interface", chat_id=chat.id))


@bp.route("/<int:chat_id>")
def chat_interface(chat_id: int) -> str | Response:
    try:
        chat_data = get_chat(chat_id)
    except DataAccessError:
        abort(400, "Invalid id.")

    recent_chats = chats_data_source.get_user_data(limit=10)

    auth = get_auth()
    assert auth.user
    is_owner = auth.user.id == chat_data.user_id
    is_current_class = auth.cur_class is not None and auth.cur_class.class_id == chat_data.class_id
    warmup_active = (
        is_owner
        and is_current_class
        and chat_data.warmup_quiz is not None
        and not chat_data.warmup_quiz.reviewed
    )
    if warmup_active:
        return render_template(
            "warmup_quiz.html",
            chat=chat_data,
            quiz=chat_data.warmup_quiz,
            quiz_kind="warmup",
            recent_chats=recent_chats,
        )

    wrapup_active = (
        is_owner
        and is_current_class
        and guided_chat_complete(chat_data)
        and chat_data.wrapup_quiz is not None
        and not chat_data.wrapup_quiz.reviewed
    )
    if wrapup_active:
        if chat_data.wrapup_quiz.started_at is None:
            chat_data.wrapup_quiz.started_at = datetime.now(UTC).isoformat()
            save_chat(chat_data)
        return render_template(
            "warmup_quiz.html",
            chat=chat_data,
            quiz=chat_data.wrapup_quiz,
            quiz_kind="wrapup",
            recent_chats=recent_chats,
        )

    show_message_input = is_owner and is_current_class

    return render_template(
        "tutor_view.html",
        chat=chat_data,
        recent_chats=recent_chats,
        msg_input=show_message_input,
        chat_complete=guided_chat_complete(chat_data),
    )


def _can_update_chat(chat: ChatData) -> bool:
    auth = get_auth()
    return (
        auth.user is not None
        and auth.cur_class is not None
        and auth.user.id == chat.user_id
        and auth.cur_class.class_id == chat.class_id
    )


@bp.route("/<int:chat_id>/warmup", methods=["POST"])
@class_enabled_required
def submit_warmup_quiz(chat_id: int) -> Response:
    return _submit_assessment_quiz(chat_id, quiz_kind="warmup")


@bp.route("/<int:chat_id>/wrapup", methods=["POST"])
@class_enabled_required
def submit_wrapup_quiz(chat_id: int) -> Response:
    return _submit_assessment_quiz(chat_id, quiz_kind="wrapup")


def _get_assessment_quiz(chat: ChatData, quiz_kind: QuizKind) -> WarmupQuiz | None:
    return chat.warmup_quiz if quiz_kind == "warmup" else chat.wrapup_quiz


def _parse_quiz_answers(quiz: WarmupQuiz) -> list[QuizAnswer] | None:
    answers: list[QuizAnswer] = []
    for index, question in enumerate(quiz.questions):
        raw_answers = request.form.getlist(f"answer_{index}")
        try:
            selected = sorted({int(raw_answer) for raw_answer in raw_answers})
        except ValueError:
            return None
        if not selected or any(not 0 <= answer < len(question.options) for answer in selected):
            return None
        if question.question_type != 'multiple_choice' and len(selected) != 1:
            return None
        answers.append(selected if question.question_type == 'multiple_choice' else selected[0])
    return answers


def _append_warmup_diagnostic(chat: ChatData, quiz: WarmupQuiz) -> None:
    assert quiz.score is not None
    objective_totals: Counter[str] = Counter()
    objective_misses: Counter[str] = Counter()
    result_lines = [
        "Before this conversation, the student completed a warm-up diagnostic on the current "
        f"tutor plan ({quiz.source_tutor_name}) and scored {quiz.score}/{len(quiz.questions)}.",
        "Prioritize objectives with missed questions. Spend more time assessing, explaining, and practicing those areas while still covering every objective:",
    ]
    for answer, question in zip(quiz.answers, quiz.questions, strict=True):
        objective = question.objective or 'Current objective'
        correct = question.is_correct(answer)
        objective_totals[objective] += 1
        if not correct:
            objective_misses[objective] += 1
        selected_indices = [answer] if isinstance(answer, int) else answer
        selected_text = ', '.join(question.options[index] for index in selected_indices)
        correct_text = ', '.join(question.options[index] for index in question.correct_answer_indices)
        result_lines.append(
            f"- {objective}: {'correct' if correct else 'incorrect'}. "
            f"Question: {question.question} Student answer: {selected_text}. Correct answer: {correct_text}."
        )

    weak_objectives = sorted(
        objective_misses,
        key=lambda objective: (-objective_misses[objective] / objective_totals[objective], objective),
    )
    if weak_objectives:
        result_lines.insert(2, "Weakest objectives, in priority order: " + "; ".join(weak_objectives))
    else:
        result_lines.insert(2, "No objective had a missed warm-up question; confirm understanding and proceed normally.")

    chat.messages[0]['content'] += "\n\n" + "\n".join(result_lines)


def _submit_assessment_quiz(chat_id: int, *, quiz_kind: QuizKind) -> Response:
    try:
        chat = get_chat(chat_id)
    except DataAccessError:
        abort(400, "Invalid id.")

    if not _can_update_chat(chat):
        abort(403)

    quiz = _get_assessment_quiz(chat, quiz_kind)
    if quiz is None:
        abort(400, f"This chat has no {quiz_kind} quiz.")
    if quiz.completed:
        return redirect(url_for("tutors.chat_interface", chat_id=chat.id))
    if quiz_kind == "wrapup" and not guided_chat_complete(chat):
        abort(409, "The wrap-up quiz is available after the tutor session is complete.")

    answers = _parse_quiz_answers(quiz)
    if answers is None:
        flash("Please answer every question before submitting.", "warning")
        return redirect(url_for("tutors.chat_interface", chat_id=chat.id))

    quiz.answers = answers
    quiz.score = sum(
        question.is_correct(answer)
        for answer, question in zip(answers, quiz.questions, strict=True)
    )
    quiz.completed = True
    quiz.completed_at = datetime.now(UTC).isoformat()
    if quiz_kind == "warmup":
        _append_warmup_diagnostic(chat, quiz)
    save_chat(chat)

    return redirect(url_for("tutors.chat_interface", chat_id=chat.id))


@bp.route("/<int:chat_id>/warmup/continue", methods=["POST"])
@class_enabled_required
def finish_warmup_quiz(chat_id: int) -> Response:
    return _finish_assessment_quiz(chat_id, quiz_kind="warmup")


@bp.route("/<int:chat_id>/wrapup/continue", methods=["POST"])
@class_enabled_required
def finish_wrapup_quiz(chat_id: int) -> Response:
    return _finish_assessment_quiz(chat_id, quiz_kind="wrapup")


def _finish_assessment_quiz(chat_id: int, *, quiz_kind: QuizKind) -> Response:
    try:
        chat = get_chat(chat_id)
    except DataAccessError:
        abort(400, "Invalid id.")

    if not _can_update_chat(chat):
        abort(403)

    quiz = _get_assessment_quiz(chat, quiz_kind)
    if quiz is None or not quiz.completed:
        abort(400, f"Complete the {quiz_kind} quiz before continuing.")

    quiz.reviewed = True
    save_chat(chat)
    return redirect(url_for("tutors.chat_interface", chat_id=chat.id))


def get_chat(chat_id: int) -> ChatData:
    chat_row = chats_data_source.get_row(chat_id)

    chat_json = chat_row['chat_json']

    try:
        chat_data = msgspec.json.decode(chat_json, type=ChatData)
    except msgspec.DecodeError as e:
        current_app.logger.error(f"Failed to decode chat {chat_id} from database. Error: {e}")
        raise DataAccessError from e

    return msgspec.structs.replace(
        chat_data,
        id=chat_id,
        user_id=chat_row['user_id'],
        user_json=chat_row['user'],
        class_id=chat_row['class_id']
    )


def save_chat(chat_data: ChatData) -> None:
    # remove redundant items (already stored elsewhere in db)
    # the Struct's omit_default=True will keep them from being encoded
    filtered = msgspec.structs.replace(
        chat_data,
        id=None,
        user_id=None,
        user_json=None,
        class_id=None
    )

    db = get_db()
    db.execute(
        "UPDATE chats SET chat_json=? WHERE id=?",
        [msgspec.json.encode(filtered).decode(), chat_data.id]
    )
    db.commit()


def erase_chat(chat_data: ChatData) -> None:
    db = get_db()
    db.execute("DELETE FROM chats WHERE id=?", [chat_data.id])
    db.commit()


def run_chat_round(llm: LLM, chat: ChatData) -> None:
    """ Run a single round of the given chat with the given LLM.

    Uses stream_chat_round and simply consumes all of its yielded outputs,
    relying on its side-effects (updating the chat in the database).
    """
    async def consume_stream_chat() -> None:
        async for _ in stream_chat_round(llm, chat):
            pass
    asyncio.run(consume_stream_chat())


async def stream_chat_round(llm: LLM, chat: ChatData) -> AsyncGenerator[str, None]:
    """ Run a single round of the given chat with the given LLM, streaming
    response chunks and potentially conversation analysis via yield.
    """
    msgs = chat.openai_messages[:]

    if len(msgs) == 0 or (len(msgs) == 1 and msgs[0]['role'] == 'system'):
        # Gemini, at least, requires a user message to start, but we don't need
        # to save or display it, so only add this to the copy of the messages
        # rather than updating the messages in the `chat` object.
        msgs.append({
            'role': 'user',
            'content': 'Please generate an initial message for the user.',
        })

    # Generate a completion and stream the response
    stream = await llm.stream_completion(messages=msgs)

    response_txt = ""
    async for chunk in stream:
        if chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta.content or ""

            if choice.finish_reason == "length":  # "length" if max_completion_tokens reached
                delta += "\n\n[error: maximum length exceeded]"

            yield delta
            response_txt += delta
        elif chunk.usage is not None:
            # usage available in the final chunk
            chat.usages.append(msgspec.convert(chunk.usage.model_dump(), Usage))

    # Update the chat w/ the response (and persist to the DB)
    chat.messages.append({
        'role': 'assistant',
        'content': response_txt,
    })
    save_chat(chat)

    if chat.mode == "guided":
        # Summarize/analyze the chat so far
        await _analyze_guided_chat(chat, llm)


async def _analyze_guided_chat(chat_data: ChatData, llm: LLM) -> ChatData:
    analyze_messages: list[ChatMessage] = [
        *chat_data.openai_messages,
        {'role': 'user', 'content': prompts.guided_analyze_tpl.render(chat=chat_data)},
    ]
    _analyze_response, analyze_response_txt = await llm.get_completion(
        messages=analyze_messages,
        extra_args={
            'response_format': {'type': 'json_object'},
        },
    )
    try:
        analysis = msgspec.json.decode(analyze_response_txt, type=GuidedAnalysis)
    except msgspec.DecodeError:
        current_app.logger.warning(f"Invalid JSON response in _analyze_guided_chat: {analyze_response_txt}")
    else:
        chat_data.analysis = analysis
        save_chat(chat_data)

    return chat_data


@bp.route("/progress/<int:chat_id>")
def get_progress(chat_id: int) -> Response:
    try:
        chat_data = get_chat(chat_id)
    except DataAccessError:
        abort(400, "Invalid id.")

    response = make_response(
        render_template(
            "progress_widget.html",
            chat=chat_data,
            chat_complete=guided_chat_complete(chat_data),
        )
    )
    chat_complete = guided_chat_complete(chat_data)
    response.headers['X-Chat-Complete'] = str(chat_complete).lower()
    response.headers['X-Wrapup-Ready'] = str(
        chat_complete
        and chat_data.wrapup_quiz is not None
        and not chat_data.wrapup_quiz.reviewed
    ).lower()
    return response


@bp.route("/post_message.sse", methods=["POST"])  # '.sse' extension needed for reverse proxy config to disable buffering / allow streaming
@class_enabled_required
@with_llm(spend_token=True)
def new_message(llm: LLM) -> Response:
    chat_id = int(request.form["id"])
    new_msg = request.form["message"]

    if len(new_msg) > MAX_MESSAGE_LEN:
        current_app.logger.warning(
            "Rejecting chat message by user %d: %d chars (max %d)",
            get_auth().user_id, len(new_msg), MAX_MESSAGE_LEN
        )
        return Response(f"Message is too long (max {MAX_MESSAGE_LEN:,} characters).", 400, mimetype='text/plain')

    # Get the specified chat
    try:
        chat = get_chat(chat_id)
    except DataAccessError:
        return Response("Invalid id.", 400, mimetype='text/plain')

    if not _can_update_chat(chat):
        return Response("You cannot update this chat.", 403, mimetype='text/plain')

    blocked_message = _chat_blocked_message(chat)
    if blocked_message is not None:
        return Response(blocked_message, 409, mimetype='text/plain')

    messages = chat.messages

    # Add the new message to the chat (persisting to the DB)
    messages.append({
        'role': 'user',
        'content': new_msg,
    })

    # Stream back a response (while "translating" the async generator into a sync generator)
    async_stream = stream_chat_round(llm, chat)

    runner = asyncio.Runner()
    try:
        # Get first chunk before we send a Response so we can catch errors
        # and abort first if needed.
        first_chunk = runner.run(anext(async_stream))
    except RuntimeError as e:
        current_app.logger.error(f"Error getting chat response: {e}")
        return Response(str(e), 502, mimetype='text/plain')

    # only save the user's new message if the response succeeds
    # (front-end can let the user re-submit it if they want)
    save_chat(chat)

    def stream_response() -> Iterator[str]:
        yield first_chunk
        while True:
            try:
                chunk: str = runner.run(anext(async_stream))
                yield chunk
            except StopAsyncIteration:
                break
        runner.close()

    return Response(
        stream_with_context(stream_response()),
        mimetype="text/event-stream",
    )
