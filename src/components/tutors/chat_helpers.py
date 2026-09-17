# SPDX-FileCopyrightText: 2026 Mark Liffiton <liffiton@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only

from datetime import UTC, datetime

import msgspec

from components.code_contexts import ContextConfig
from gened.auth import get_auth
from gened.db import get_db
from gened.llm import LLM, ChatMessage

from . import prompts
from .data_types import (
    ChatData,
    ChatMode,
    GuidedAnalysis,
    GuidedObjectiveProgress,
    QuizKind,
    QuizQuestionType,
    TutorConfig,
    WarmupQuiz,
    WarmupQuizQuestion,
    WarmupQuizResponse,
)

DEFAULT_WARMUP_QUESTIONS = 10
MIN_QUIZ_OPTIONS = 2
MAX_QUIZ_OPTIONS = 5
QUIZ_TYPES: frozenset[QuizQuestionType] = frozenset({'single_choice', 'multiple_choice', 'true_false'})


class WarmupQuizGenerationError(ValueError):
    """Raised when an LLM response is not a usable warm-up quiz."""


def create_guided_chat(
    tutor_config: TutorConfig,
    *,
    warmup_quiz: WarmupQuiz | None = None,
    wrapup_quiz: WarmupQuiz | None = None,
    skip_db: bool = False,
) -> ChatData:
    auth = get_auth()

    # Get documents marked for use in chat
    chat_docs = [doc for doc in tutor_config.documents if 'chat' in doc.use_in]

    tikz_enabled = 'tikz_experiment' in auth.class_experiments
    sys_prompt = prompts.guided_sys_msg_tpl.render(tutor_config=tutor_config, documents=chat_docs, tikz_enabled=tikz_enabled)

    chat = _create_chat(tutor_config.topic, context_name=None, sys_prompt=sys_prompt, mode="guided")

    chat.analysis = GuidedAnalysis(
        summary = "",
        progress = [
            GuidedObjectiveProgress(obj.name, "not started")
            for obj in tutor_config.objectives
        ],
    )
    chat.warmup_quiz = warmup_quiz
    chat.wrapup_quiz = wrapup_quiz

    if not skip_db:
        chat = _save_chat(chat)

    return chat


def _validate_quiz_questions(
    questions: list[WarmupQuizQuestion],
    tutor_config: TutorConfig,
    *,
    num_questions: int,
    quiz_kind: QuizKind,
) -> None:
    if len(questions) != num_questions:
        reason = f"Expected {num_questions} {quiz_kind} questions, got {len(questions)}"
        raise WarmupQuizGenerationError(reason)

    objective_names = {objective.name for objective in tutor_config.objectives}
    seen_types: set[QuizQuestionType] = set()
    for question in questions:
        if not question.question.strip():
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} question text cannot be empty")
        if not MIN_QUIZ_OPTIONS <= len(question.options) <= MAX_QUIZ_OPTIONS:
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} questions must have 2 to 5 options")
        normalized_options = [option.strip().casefold() for option in question.options]
        if any(not option for option in normalized_options) or len(set(normalized_options)) != len(normalized_options):
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} question options must be non-empty and distinct")

        correct_indices = question.correct_answer_indices
        if not correct_indices or any(not 0 <= index < len(question.options) for index in correct_indices):
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} correct indices are outside the options list")
        if question.question_type in {'single_choice', 'true_false'} and len(correct_indices) != 1:
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} single-answer questions need exactly one correct option")
        if question.question_type == 'multiple_choice' and not 2 <= len(correct_indices) < len(question.options):
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} multiple-choice questions need multiple correct and at least one incorrect option")
        if question.question_type == 'true_false' and normalized_options != ['true', 'false']:
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} true/false questions must use True and False options")
        if question.objective not in objective_names:
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} question references an unknown learning objective")
        if not question.explanation.strip():
            raise WarmupQuizGenerationError(f"{quiz_kind.title()} explanations cannot be empty")
        seen_types.add(question.question_type)

    if seen_types != QUIZ_TYPES:
        missing = ', '.join(sorted(QUIZ_TYPES - seen_types))
        raise WarmupQuizGenerationError(f"{quiz_kind.title()} quiz is missing required question types: {missing}")


async def generate_assessment_quizzes(
    tutor_config: TutorConfig,
    llm: LLM,
    num_questions: int = DEFAULT_WARMUP_QUESTIONS,
) -> tuple[list[WarmupQuizQuestion], list[WarmupQuizQuestion]]:
    """Generate and validate paired assessments for the current tutor plan."""
    async def generate_one(
        quiz_kind: QuizKind,
        warmup_questions: list[WarmupQuizQuestion],
    ) -> list[WarmupQuizQuestion]:
        messages: list[ChatMessage] = [
            {
                'role': 'system',
                'content': prompts.assessment_quiz_sys_prompt.render(
                    tutor_config=tutor_config,
                    num_questions=num_questions,
                    quiz_kind=quiz_kind,
                    warmup_questions=warmup_questions,
                ),
            },
            {
                'role': 'user',
                'content': f"Generate the {num_questions}-question {quiz_kind} assessment now.",
            },
        ]
        _response, response_txt = await llm.get_completion(
            messages=messages,
            extra_args={
                'response_format': {'type': 'json_object'},
                # Ten mixed-format questions with explanations can exceed the
                # low default output cap used by some OpenAI-compatible APIs.
                'max_completion_tokens': 8192,
            },
        )
        response = msgspec.json.decode(response_txt, type=WarmupQuizResponse)
        _validate_quiz_questions(response.questions, tutor_config, num_questions=num_questions, quiz_kind=quiz_kind)
        return response.questions

    warmup_questions = await generate_one('warmup', [])
    wrapup_questions = await generate_one('wrapup', warmup_questions)

    warmup_text = {question.question.strip().casefold() for question in warmup_questions}
    wrapup_text = {question.question.strip().casefold() for question in wrapup_questions}
    if warmup_text & wrapup_text:
        raise WarmupQuizGenerationError("Warm-up and wrap-up questions must use different wording")

    return warmup_questions, wrapup_questions


async def get_or_create_assessment_quizzes(
    tutor_config: TutorConfig,
    llm: LLM,
    num_questions: int = DEFAULT_WARMUP_QUESTIONS,
) -> tuple[WarmupQuiz, WarmupQuiz]:
    """Return the shared quiz pair, generating it once for this tutor config."""
    stored_config_json: str | None = None
    if tutor_config.row_id is not None and not (tutor_config.warmup_questions and tutor_config.wrapup_questions):
        row = get_db().execute(
            "SELECT * FROM config_items WHERE id=? AND item_type='guided_tutor'",
            [tutor_config.row_id],
        ).fetchone()
        if row is not None:
            stored_config_json = row['config']
            current_config = TutorConfig.from_row(row)
            if current_config.warmup_questions and current_config.wrapup_questions:
                tutor_config.warmup_questions = current_config.warmup_questions
                tutor_config.wrapup_questions = current_config.wrapup_questions

    if tutor_config.warmup_questions and tutor_config.wrapup_questions:
        warmup_questions = tutor_config.warmup_questions
        wrapup_questions = tutor_config.wrapup_questions
        _validate_quiz_questions(warmup_questions, tutor_config, num_questions=num_questions, quiz_kind='warmup')
        _validate_quiz_questions(wrapup_questions, tutor_config, num_questions=num_questions, quiz_kind='wrapup')
    else:
        warmup_questions, wrapup_questions = await generate_assessment_quizzes(tutor_config, llm, num_questions)
        tutor_config.warmup_questions = warmup_questions
        tutor_config.wrapup_questions = wrapup_questions

        if tutor_config.row_id is not None:
            cursor = get_db().execute(
                "UPDATE config_items SET config=? WHERE id=? AND item_type='guided_tutor' AND config=?",
                [tutor_config.to_json(), tutor_config.row_id, stored_config_json],
            )
            get_db().commit()
            if cursor.rowcount == 0:
                row = get_db().execute(
                    "SELECT * FROM config_items WHERE id=? AND item_type='guided_tutor'",
                    [tutor_config.row_id],
                ).fetchone()
                if row is None:
                    raise WarmupQuizGenerationError("Tutor configuration was removed while assessments were generated")
                current_config = TutorConfig.from_row(row)
                if not (current_config.warmup_questions and current_config.wrapup_questions):
                    raise WarmupQuizGenerationError("Tutor configuration changed while assessments were generated; please retry")
                warmup_questions = current_config.warmup_questions
                wrapup_questions = current_config.wrapup_questions

    started_at = datetime.now(UTC).isoformat()
    warmup = WarmupQuiz(
        source_tutor_name=tutor_config.name,
        questions=warmup_questions,
        quiz_kind='warmup',
        started_at=started_at,
    )
    wrapup = WarmupQuiz(
        source_tutor_name=tutor_config.name,
        questions=wrapup_questions,
        quiz_kind='wrapup',
    )
    return warmup, wrapup


async def generate_warmup_quiz(
    tutor_config: TutorConfig,
    llm: LLM,
    num_questions: int = DEFAULT_WARMUP_QUESTIONS,
) -> WarmupQuiz:
    """Compatibility wrapper returning the current-session warm-up."""
    warmup, _wrapup = await get_or_create_assessment_quizzes(
        tutor_config,
        llm,
        num_questions,
    )
    return warmup


def create_inquiry_chat(topic: str, context: ContextConfig | None, *, skip_db: bool = False) -> ChatData:
    auth = get_auth()

    context_name = context.name if context else None
    context_string = context.prompt_str() if context else None

    tikz_enabled = 'tikz_experiment' in auth.class_experiments
    sys_prompt = prompts.inquiry_sys_msg_tpl.render(topic=topic, context=context_string, tikz_enabled=tikz_enabled)

    chat = _create_chat(topic, context_name, sys_prompt, "inquiry")

    if not skip_db:
        chat = _save_chat(chat)

    return chat


def _create_chat(topic: str, context_name: str | None, sys_prompt: str, mode: ChatMode) -> ChatData:
    auth = get_auth()
    user_id = auth.user_id
    class_id = auth.cur_class.class_id if auth.cur_class else None

    chat_data = ChatData(
        user_id=user_id,
        class_id=class_id,
        topic=topic,
        context_name=context_name,
        messages=[{"role": "system", "content": sys_prompt}],
        usages=[],
        mode=mode,
    )

    return chat_data


def _save_chat(chat_data: ChatData) -> ChatData:
    """ Record the given chat in the database, updating its id attribute with the new row id. """
    auth = get_auth()
    user_id = auth.user_id
    role_id = auth.cur_class.role_id if auth.cur_class else None

    db = get_db()
    cur = db.execute(
        "INSERT INTO chats (user_id, role_id, chat_json) VALUES (?, ?, ?)",
        [user_id, role_id, msgspec.json.encode(chat_data).decode()]
    )
    new_row_id = cur.lastrowid

    db.commit()

    assert new_row_id is not None
    return msgspec.structs.replace(chat_data, id=new_row_id)
