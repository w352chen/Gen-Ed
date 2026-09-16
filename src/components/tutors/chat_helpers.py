# SPDX-FileCopyrightText: 2026 Mark Liffiton <liffiton@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only

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
    TutorConfig,
    WarmupQuiz,
    WarmupQuizResponse,
)

DEFAULT_WARMUP_QUESTIONS = 10
MIN_QUIZ_OPTIONS = 2
MAX_QUIZ_OPTIONS = 4


class WarmupQuizGenerationError(ValueError):
    """Raised when an LLM response is not a usable warm-up quiz."""


def create_guided_chat(
    tutor_config: TutorConfig,
    *,
    warmup_quiz: WarmupQuiz | None = None,
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

    if not skip_db:
        chat = _save_chat(chat)

    return chat


async def generate_warmup_quiz(
    tutor_config: TutorConfig,
    llm: LLM,
    num_questions: int = DEFAULT_WARMUP_QUESTIONS,
) -> WarmupQuiz:
    """Generate and validate a quiz from the preceding tutor plan."""
    messages: list[ChatMessage] = [
        {
            'role': 'system',
            'content': prompts.warmup_quiz_sys_prompt.render(
                tutor_config=tutor_config,
                num_questions=num_questions,
            ),
        },
        {
            'role': 'user',
            'content': f"Generate the {num_questions}-question warm-up quiz now.",
        },
    ]
    _response, response_txt = await llm.get_completion(
        messages=messages,
        extra_args={'response_format': {'type': 'json_object'}},
    )
    response = msgspec.json.decode(response_txt, type=WarmupQuizResponse)

    if len(response.questions) != num_questions:
        reason = f"Expected {num_questions} warm-up questions, got {len(response.questions)}"
        raise WarmupQuizGenerationError(reason)

    for question in response.questions:
        if not question.question.strip():
            reason = "Warm-up question text cannot be empty"
            raise WarmupQuizGenerationError(reason)
        if not MIN_QUIZ_OPTIONS <= len(question.options) <= MAX_QUIZ_OPTIONS:
            reason = "Warm-up questions must have 2 to 4 options"
            raise WarmupQuizGenerationError(reason)
        normalized_options = [option.strip().casefold() for option in question.options]
        if any(not option for option in normalized_options) or len(set(normalized_options)) != len(normalized_options):
            reason = "Warm-up question options must be non-empty and distinct"
            raise WarmupQuizGenerationError(reason)
        if not 0 <= question.correct_index < len(question.options):
            reason = "Warm-up correct_index is outside the options list"
            raise WarmupQuizGenerationError(reason)
        if not question.explanation.strip():
            reason = "Warm-up explanations cannot be empty"
            raise WarmupQuizGenerationError(reason)

    return WarmupQuiz(
        source_tutor_name=tutor_config.name,
        questions=response.questions,
    )


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
