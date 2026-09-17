# SPDX-FileCopyrightText: 2026 Mark Liffiton <liffiton@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only

"""Tests for tutors component serialization and data models."""

import asyncio
import re
from typing import Any

import msgspec
import pytest
from flask import Flask
from werkzeug.datastructures import ImmutableMultiDict, MultiDict

from components.tutors.chat import guided_chat_complete
from components.tutors.chat_helpers import get_or_create_assessment_quizzes
from components.tutors.data import fmt_analysis
from components.tutors.data_types import (
    ChatData,
    ContextDocument,
    GuidedAnalysis,
    GuidedObjectiveProgress,
    LearningObjective,
    TutorConfig,
    WarmupQuiz,
    WarmupQuizQuestion,
)
from gened.db import get_db
from tests.conftest import AppClient


def assessment_questions(prefix: str, objective: str) -> list[WarmupQuizQuestion]:
    """Build a valid 10-question mixed-format assessment for tests."""
    questions = [
        WarmupQuizQuestion(
            question=f"{prefix} single choice",
            options=["Correct", "Incorrect"],
            correct_indices=[0],
            explanation="Single-choice explanation.",
            objective=objective,
            question_type="single_choice",
        ),
        WarmupQuizQuestion(
            question=f"{prefix} multiple choice",
            options=["Correct A", "Correct B", "Incorrect"],
            correct_indices=[0, 1],
            explanation="Multiple-choice explanation.",
            objective=objective,
            question_type="multiple_choice",
        ),
        WarmupQuizQuestion(
            question=f"{prefix} true or false",
            options=["True", "False"],
            correct_indices=[0],
            explanation="True/false explanation.",
            objective=objective,
            question_type="true_false",
        ),
    ]
    questions.extend(
        WarmupQuizQuestion(
            question=f"{prefix} single choice {index}",
            options=["Correct", "Incorrect"],
            correct_indices=[0],
            explanation="Single-choice explanation.",
            objective=objective,
            question_type="single_choice",
        )
        for index in range(4, 11)
    )
    return questions


def correct_quiz_form(questions: list[WarmupQuizQuestion], *, first_wrong: bool = False) -> MultiDict[str, str]:
    data: MultiDict[str, str] = MultiDict()
    for index, question in enumerate(questions):
        correct_indices = question.correct_answer_indices
        if first_wrong and index == 0:
            correct_indices = [1]
        for answer in correct_indices:
            data.add(f"answer_{index}", str(answer))
    return data


def test_chat_data_roundtrip() -> None:
    """Test serializing and deserializing ChatData with various edge cases."""
    # Test with all fields including complex nested structures
    original = ChatData(
        id=1,
        user_id=11,
        user_json='{"display_name": "Test User"}',
        class_id=5,
        topic="Test topic with special chars: <>&'",
        context_name="ctx1",
        messages=[
            {'role': 'system', 'content': 'You are a tutor'},
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ],
        usages=[{'prompt_tokens': 10, 'completion_tokens': 20}, {'prompt_tokens': 15, 'completion_tokens': 25}],
        mode="guided",
        analysis=GuidedAnalysis(
            summary='test',
            progress=[GuidedObjectiveProgress(objective='Vars', status='completed')]
        )
    )
    # Serialize
    json_bytes = msgspec.json.encode(original)
    # Deserialize
    restored = msgspec.json.decode(json_bytes, type=ChatData)
    # Verify all fields match
    assert restored == original
    assert restored.topic == original.topic
    assert restored.messages == original.messages
    assert restored.usages == original.usages
    assert restored.analysis == original.analysis

    # Test with empty strings and special values
    minimal = ChatData(
        id=2,
        user_id=12,
        topic="",
        messages=[],
        mode="inquiry"
    )
    json_bytes2 = msgspec.json.encode(minimal)
    restored2 = msgspec.json.decode(json_bytes2, type=ChatData)
    assert restored2 == minimal
    assert restored2.topic == ""
    assert restored2.messages == []


def test_chat_data_roundtrip_with_warmup_quiz() -> None:
    quiz = WarmupQuiz(
        source_tutor_name="Week 1",
        questions=[
            WarmupQuizQuestion(
                question="Python lists are mutable.",
                options=["True", "False"],
                correct_index=0,
                explanation="List contents can be changed after creation.",
                objective="Explain list mutability",
            )
        ],
        answers=[0],
        score=1,
        completed=True,
    )
    wrapup = WarmupQuiz(
        source_tutor_name="Week 1",
        quiz_kind="wrapup",
        questions=[
            WarmupQuizQuestion(
                question="Select both mutable collections.",
                options=["list", "dict", "tuple"],
                correct_indices=[0, 1],
                explanation="Lists and dictionaries are mutable.",
                objective="Explain mutability",
                question_type="multiple_choice",
            )
        ],
        answers=[[0, 1]],
        score=1,
        completed=True,
    )
    original = ChatData(
        topic="Week 1",
        messages=[{'role': 'system', 'content': 'Tutor instructions'}],
        mode="guided",
        warmup_quiz=quiz,
        wrapup_quiz=wrapup,
    )

    restored = msgspec.json.decode(msgspec.json.encode(original), type=ChatData)

    assert restored == original
    assert restored.warmup_quiz is not None
    assert restored.warmup_quiz.score == 1
    assert not restored.warmup_quiz.reviewed
    assert restored.wrapup_quiz is not None
    assert restored.wrapup_quiz.answers == [[0, 1]]


def test_tutor_config_roundtrip() -> None:
    """Test serializing and deserializing TutorConfig."""
    original = TutorConfig(
        name="Python Basics",
        topic="Introduction to Python",
        context="Learning context",
        documents=[ContextDocument(filename="notes.txt", text="Some notes", use_in={"setup", "chat"})],
        objectives=[
            LearningObjective(name="Variables", questions=["What is a variable?", "How to use variables?"]),
            LearningObjective(name="Functions", questions=["How do you define a function?"])
        ]
    )
    # Serialize using to_json (from ConfigItem)
    json_str = original.to_json()
    # Deserialize
    data = msgspec.json.decode(json_str)
    data['name'] = "Python Basics"
    # Convert dict objectives to LearningObjective objects (as from_row does)
    if data.get('objectives'):
        data['objectives'] = [msgspec.convert(obj, LearningObjective) for obj in data['objectives']]
    restored = msgspec.convert(data, TutorConfig)
    # Verify all fields match
    assert restored == original
    assert len(restored.documents) == len(original.documents)
    assert restored.documents[0].filename == original.documents[0].filename
    assert restored.documents[0].text == original.documents[0].text
    assert restored.documents[0].use_in == original.documents[0].use_in

    # Test with empty objectives
    minimal = TutorConfig(name="Minimal", topic="Test", objectives=[])
    json_str2 = minimal.to_json()
    data2 = msgspec.json.decode(json_str2)
    data2['name'] = "Minimal"
    data2['objectives'] = []
    restored2 = msgspec.convert(data2, TutorConfig)
    assert restored2 == minimal


def test_tutor_config_from_request_form() -> None:
    """Test creating TutorConfig from request form."""
    form: ImmutableMultiDict[str, Any] = ImmutableMultiDict([
        ('name', 'Python Basics'),
        ('topic', 'Introduction to Python'),
        ('context', 'Learning context'),
        ('document_filename[]', 'notes.txt'),
        ('document_text[]', 'Some notes about Python'),
        ('document_use_in[]', 'setup,chat'),
        ('objectives', 'Variables'),
        ('objectives', 'Functions'),
        ('questions[0]', 'What is a variable?'),
        ('questions[0]', 'How do you use variables?'),
        ('questions[1]', 'How do you define a function?'),
    ])

    config = TutorConfig.from_request_form(form)

    assert config.name == "Python Basics"
    assert config.topic == "Introduction to Python"
    assert len(config.documents) == 1
    assert config.documents[0].filename == "notes.txt"
    assert config.documents[0].text == "Some notes about Python"
    assert config.documents[0].use_in == {'setup', 'chat'}
    assert len(config.objectives) == 2
    assert config.objectives[0].name == "Variables"
    assert config.objectives[0].questions == ['What is a variable?', 'How do you use variables?']
    assert config.objectives[1].name == "Functions"
    assert config.objectives[1].questions == ['How do you define a function?']


def test_read_chat_from_database(client: AppClient) -> None:
    """Test reading chat data from the database via HTTP request."""
    # Login to set up auth context
    client.login('testuser', 'testpassword')
    # Chat ID 1 exists in test_data.sql
    # Access it via the chat interface
    response = client.get('/tutor/1')
    assert response.status_code == 200
    # Verify the chat content is rendered
    assert 'topic1' in response.text
    assert 'user_msg_1' in response.text
    assert 'assistant_msg_1' in response.text


def test_read_chat_with_analysis_from_database(app: Flask, client: AppClient) -> None:
    """Test reading chat with analysis data from database."""
    # Insert a chat with analysis data
    with app.app_context():
        db = get_db()
        chat_json = {
            "topic": "Python Basics",
            "mode": "guided",
            "messages": [
                {"role": "system", "content": "You are a tutor"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ],
            "usages": [{"prompt_tokens": 10, "completion_tokens": 20}],
            "analysis": {
                "summary": "Student progressing well",
                "progress": [
                    {"objective": "Variables", "status": "completed"}
                ]
            }
        }
        db.execute(
            "INSERT INTO chats (chat_json, user_id, role_id) VALUES (?, ?, ?)",
            [msgspec.json.encode(chat_json).decode(), 11, 4]
        )
        db.commit()
        new_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Access via HTTP request
    client.login('testuser', 'testpassword')
    response = client.get(f'/tutor/{new_id}')
    assert response.status_code == 200
    assert 'Python Basics' in response.text
    # Verify analysis data is present
    assert 'Variables' in response.text
    assert 'completed' in response.text


def test_completed_guided_chat_is_locked(app: Flask, client: AppClient) -> None:
    completed_chat = ChatData(
        topic="Completed tutor",
        mode="guided",
        messages=[
            {'role': 'user', 'content': 'Finished'},
            {'role': 'assistant', 'content': 'Well done'},
        ],
        analysis=GuidedAnalysis(
            summary="All objectives addressed",
            progress=[
                GuidedObjectiveProgress(objective="Variables", status="completed"),
                GuidedObjectiveProgress(objective="Loops", status="moved on"),
            ],
        ),
    )
    assert guided_chat_complete(completed_chat)

    incomplete_chat = msgspec.structs.replace(
        completed_chat,
        analysis=GuidedAnalysis(
            summary="Still working",
            progress=[GuidedObjectiveProgress(objective="Variables", status="in progress")],
        ),
    )
    assert not guided_chat_complete(incomplete_chat)

    with app.app_context():
        cursor = get_db().execute(
            "INSERT INTO chats (chat_json, user_id, role_id) VALUES (?, ?, ?)",
            [msgspec.json.encode(completed_chat).decode(), 11, 4],
        )
        get_db().commit()
        chat_id = cursor.lastrowid
        assert chat_id is not None

    client.login('testuser', 'testpassword')
    client.get('/classes/switch/2')

    response = client.get(f'/tutor/{chat_id}')
    assert response.status_code == 200
    assert 'data-chat-complete="true"' in response.text
    assert 'Tutor session complete. All learning objectives have been addressed.' in response.text
    assert re.search(r'<textarea[^>]+disabled', response.text)
    assert re.search(r'<button[^>]+disabled[^>]*>\s*Send', response.text)

    progress_response = client.get(f'/tutor/progress/{chat_id}')
    assert progress_response.status_code == 200
    assert progress_response.headers['X-Chat-Complete'] == 'true'

    post_response = client.post(
        '/tutor/post_message.sse',
        data={'id': chat_id, 'message': 'One more message'},
    )
    assert post_response.status_code == 409
    assert post_response.text == 'This tutor session is complete.'

    with app.app_context():
        saved_json = get_db().execute(
            "SELECT chat_json FROM chats WHERE id=?",
            [chat_id],
        ).fetchone()['chat_json']
        saved_chat = msgspec.json.decode(saved_json, type=ChatData)
        assert [message['content'] for message in saved_chat.messages] == ['Finished', 'Well done']


def test_read_chat_with_null_usages_from_database(app: Flask, client: AppClient) -> None:
    """A chat whose stored usages contain nulls (written by an older bug) must
    still load rather than erroring out."""
    # Insert a chat with nulls in the usages, as the old code would have saved
    with app.app_context():
        db = get_db()
        chat_json = {
            "topic": "Corrupt usages",
            "mode": "guided",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ],
            "usages": [
                {
                    "completion_tokens": 12,
                    "prompt_tokens": 456,
                    "total_tokens": 468,
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": None,
                        "audio_tokens": None,
                        "reasoning_tokens": None,
                        "rejected_prediction_tokens": None,
                    },
                    "prompt_tokens_details": {
                        "audio_tokens": None,
                        "cache_write_tokens": None,
                        "cached_tokens": None,
                    },
                }
            ],
        }
        db.execute(
            "INSERT INTO chats (chat_json, user_id, role_id) VALUES (?, ?, ?)",
            [msgspec.json.encode(chat_json).decode(), 11, 4]
        )
        db.commit()
        new_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Access via HTTP request
    client.login('testuser', 'testpassword')
    response = client.get(f'/tutor/{new_id}')
    assert response.status_code == 200
    assert 'Corrupt usages' in response.text
    assert 'Hello' in response.text


def test_fmt_analysis_valid() -> None:
    """Test fmt_analysis with valid analysis JSON."""
    analysis_json = '''{
        "summary": "Student is progressing well",
        "progress": [
            {"objective": "Variables", "status": "completed"},
            {"objective": "Functions", "status": "completed"},
            {"objective": "Loops", "status": "in progress"},
            {"objective": "Recursion", "status": "not started"}
        ]
    }'''
    result = fmt_analysis(analysis_json)
    # Should contain tags for each status
    result_str = str(result)
    assert "tag" in result_str
    # Should not contain "parse error"
    assert "parse error" not in result_str


def test_fmt_analysis_invalid_json(app: Flask) -> None:
    """Test fmt_analysis with invalid JSON."""
    invalid_json = 'not valid json'
    with app.app_context():
        assert fmt_analysis(invalid_json) == "parse error"

    invalid_json = '{"summary": "test"}'
    with app.app_context():
        assert fmt_analysis(invalid_json) == "parse error"


def test_chat_save_and_retrieve(app: Flask, client: AppClient) -> None:
    """Test saving a ChatData object and retrieving it."""
    # Login to set up auth context
    client.login('testuser', 'testpassword')

    # Create and save a ChatData object
    original = ChatData(
        user_id=11,
        topic="Test Save and Retrieve",
        mode="inquiry",
        messages=[
            {"role": "system", "content": "You are a helpful tutor"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
    )

    # Insert into database
    with app.app_context():
        db = get_db()
        db.execute(
            "INSERT INTO chats (chat_json, user_id, role_id) VALUES (?, ?, ?)",
            [msgspec.json.encode(original).decode(), 11, 4]
        )
        db.commit()
        new_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        original.id = new_id

    # Retrieve via HTTP request
    response = client.get(f'/tutor/{new_id}')
    assert response.status_code == 200
    assert 'Test Save and Retrieve' in response.text
    assert 'You are a helpful tutor' not in response.text
    assert 'Hello' in response.text
    assert 'Hi there' in response.text


def test_guided_chat_warmup_quiz_flow(
    app: Flask,
    client: AppClient,
) -> None:
    objective = "Define and call a function"
    warmup_questions = assessment_questions("Warm-up", objective)
    wrapup_questions = assessment_questions("Wrap-up", objective)
    current = TutorConfig(
        name="Week 2: Functions",
        topic="Python functions",
        context="An introductory Python course",
        objectives=[LearningObjective(name=objective)],
        opening_message="Welcome to week 2",
        warmup_questions=warmup_questions,
        wrapup_questions=wrapup_questions,
    )

    with app.app_context():
        db = get_db()
        cursor = db.execute(
            """
            INSERT INTO config_items (class_id, item_type, name, class_order, available, config)
            VALUES (2, 'guided_tutor', ?, 10, '0001-01-01', ?)
            """,
            [current.name, current.to_json()],
        )
        current_id = cursor.lastrowid
        db.commit()

    client.login('testuser', 'testpassword')
    client.get('/classes/switch/2')
    response = client.post('/tutor/new/guided', data={'tutor_id': current_id})
    assert response.status_code == 302
    chat_url = response.headers['Location']

    response = client.get(chat_url)
    assert response.status_code == 200
    assert 'Warm-up Quiz' in response.text
    assert current.name in response.text
    assert 'Select all that apply' in response.text
    assert 'True or false' in response.text
    assert 'Question 10 of 10' in response.text

    chat_id = int(chat_url.rstrip('/').rsplit('/', 1)[1])
    response = client.post('/tutor/post_message.sse', data={'id': chat_id, 'message': 'Skip the quiz'})
    assert response.status_code == 409

    response = client.post(
        f'/tutor/{chat_id}/warmup',
        data=correct_quiz_form(warmup_questions, first_wrong=True),
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert 'Your score: 9 / 10' in response.text
    assert 'Correct answer' in response.text

    response = client.post(f'/tutor/{chat_id}/warmup/continue', follow_redirects=True)
    assert response.status_code == 200
    assert 'Warm-up Quiz' not in response.text
    assert 'Welcome to week 2' in response.text

    with app.app_context():
        chat_json = get_db().execute("SELECT chat_json FROM chats WHERE id=?", [chat_id]).fetchone()['chat_json']
        saved_chat = msgspec.json.decode(chat_json, type=ChatData)
        assert saved_chat.warmup_quiz is not None
        assert saved_chat.warmup_quiz.reviewed
        assert saved_chat.warmup_quiz.completed_at is not None
        assert "current tutor plan" in saved_chat.messages[0]['content']
        assert "scored 9/10" in saved_chat.messages[0]['content']
        assert "Weakest objectives" in saved_chat.messages[0]['content']

        assert saved_chat.analysis is not None
        saved_chat.analysis = GuidedAnalysis(
            summary="All objectives addressed",
            progress=[GuidedObjectiveProgress(objective=objective, status="completed")],
        )
        get_db().execute(
            "UPDATE chats SET chat_json=? WHERE id=?",
            [msgspec.json.encode(saved_chat).decode(), chat_id],
        )
        get_db().commit()

    response = client.get(chat_url)
    assert response.status_code == 200
    assert 'Wrap-up Quiz' in response.text
    assert 'Question 10 of 10' in response.text

    response = client.post(
        f'/tutor/{chat_id}/wrapup',
        data=correct_quiz_form(wrapup_questions),
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert 'Your score: 10 / 10' in response.text
    assert 'Learning gain: +1 points' in response.text

    response = client.post(f'/tutor/{chat_id}/wrapup/continue', follow_redirects=True)
    assert response.status_code == 200
    assert 'Assessment results:' in response.text
    assert 'Warm-up 9/10' in response.text
    assert 'Wrap-up 10/10' in response.text
    assert 'Learning gain +1' in response.text

    with app.app_context():
        chat_json = get_db().execute("SELECT chat_json FROM chats WHERE id=?", [chat_id]).fetchone()['chat_json']
        saved_chat = msgspec.json.decode(chat_json, type=ChatData)
        assert saved_chat.wrapup_quiz is not None
        assert saved_chat.wrapup_quiz.reviewed
        assert saved_chat.wrapup_quiz.completed_at is not None
        assert saved_chat.wrapup_quiz.answers[1] == [0, 1]


def test_first_guided_tutor_uses_current_session_warmup(
    app: Flask,
    client: AppClient,
) -> None:
    objective = "Recognize the core terminology"
    first_tutor = TutorConfig(
        name="Week 1",
        topic="Course introduction",
        context="An introductory course",
        objectives=[LearningObjective(name=objective)],
        opening_message="Welcome to the first week",
        warmup_questions=assessment_questions("Week 1 warm-up", objective),
        wrapup_questions=assessment_questions("Week 1 wrap-up", objective),
    )
    with app.app_context():
        cursor = get_db().execute(
            """
            INSERT INTO config_items (class_id, item_type, name, class_order, available, config)
            VALUES (2, 'guided_tutor', ?, 10, '0001-01-01', ?)
            """,
            [first_tutor.name, first_tutor.to_json()],
        )
        tutor_id = cursor.lastrowid
        get_db().commit()

    client.login('testuser', 'testpassword')
    client.get('/classes/switch/2')

    response = client.post('/tutor/new/guided', data={'tutor_id': tutor_id}, follow_redirects=True)

    assert response.status_code == 200
    assert 'Warm-up Quiz' in response.text
    assert 'Week 1 warm-up single choice' in response.text
    assert 'previous' not in response.text.lower()


def test_assessment_quizzes_are_generated_once_and_reused(
    app: Flask,
) -> None:
    objective = "Use Python functions"
    tutor = TutorConfig(
        name="Functions",
        topic="Python functions",
        objectives=[LearningObjective(name=objective)],
    )
    response_payloads = [
        {"questions": msgspec.to_builtins(assessment_questions("Shared warm-up", objective))},
        {"questions": msgspec.to_builtins(assessment_questions("Shared wrap-up", objective))},
    ]

    class FakeLLM:
        calls = 0

        async def get_completion(self, **_kwargs: Any) -> tuple[dict[str, Any], str]:
            self.calls += 1
            return {}, msgspec.json.encode(response_payloads[self.calls - 1]).decode()

    fake_llm: Any = FakeLLM()
    with app.app_context():
        cursor = get_db().execute(
            """
            INSERT INTO config_items (class_id, item_type, name, class_order, available, config)
            VALUES (2, 'guided_tutor', ?, 10, '0001-01-01', ?)
            """,
            [tutor.name, tutor.to_json()],
        )
        tutor.row_id = cursor.lastrowid
        get_db().commit()

        first_warmup, first_wrapup = asyncio.run(get_or_create_assessment_quizzes(tutor, fake_llm))
        assert fake_llm.calls == 2

        row = get_db().execute("SELECT * FROM config_items WHERE id=?", [tutor.row_id]).fetchone()
        reloaded = TutorConfig.from_row(row)
        second_warmup, second_wrapup = asyncio.run(get_or_create_assessment_quizzes(reloaded, fake_llm))

    assert fake_llm.calls == 2
    assert first_warmup.questions == second_warmup.questions
    assert first_wrapup.questions == second_wrapup.questions
