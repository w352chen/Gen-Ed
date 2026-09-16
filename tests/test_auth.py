# SPDX-FileCopyrightText: 2023 Mark Liffiton <liffiton@gmail.com>
#
# SPDX-License-Identifier: AGPL-3.0-only

import re
from dataclasses import dataclass

import pytest
from flask import Flask, url_for
from flask.testing import FlaskCliRunner
from werkzeug.security import check_password_hash
from werkzeug.test import TestResponse

from gened.auth import get_auth
from gened.db import get_db
from tests.conftest import AppClient


def test_login_page(client: AppClient) -> None:
    response = client.get('/auth/login')
    assert response.status_code == 200
    assert "Username:" in response.text
    assert "Password:" in response.text
    assert 'name="username"' in response.text
    assert 'name="password"' in response.text
    assert 'type="submit"' in response.text
    assert 'Create a local account' in response.text


def _registration_csrf(client: AppClient, next_url: str = '') -> str:
    response = client.get('/auth/register', query_string={'next': next_url})
    assert response.status_code == 200
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
    assert match
    return match.group(1)


def test_public_local_registration(app: Flask, client: AppClient) -> None:
    csrf_token = _registration_csrf(client, '/profile/')
    response = client.post(
        '/auth/register',
        data={
            'csrf_token': csrf_token,
            'next': '/profile/',
            'username': 'New.User',
            'password': 'correct horse battery staple',
            'password_confirm': 'correct horse battery staple',
        },
    )

    assert response.status_code == 302
    assert response.location == '/profile/'

    with app.app_context():
        row = get_db().execute(
            """
            SELECT users.auth_name, users.is_admin, users.is_tester,
                   users.query_tokens, auth_local.username, auth_local.password
            FROM users
            JOIN auth_local ON auth_local.user_id=users.id
            WHERE auth_local.username=?
            """,
            ['new.user'],
        ).fetchone()
        assert row is not None
        assert row['auth_name'] == 'new.user'
        assert row['username'] == 'new.user'
        assert not row['is_admin']
        assert not row['is_tester']
        assert row['query_tokens'] == 0
        assert row['password'] != 'correct horse battery staple'
        assert check_password_hash(row['password'], 'correct horse battery staple')

    with client:
        profile_response = client.get("/profile/")
        assert profile_response.status_code == 200
        auth = get_auth()
        assert auth.user is not None
        assert auth.user.display_name == 'new.user'


@pytest.mark.parametrize(
    ('username', 'password', 'password_confirm', 'message'),
    [
        ('bad name', 'correct horse battery staple', 'correct horse battery staple', 'Username must be'),
        ('valid-name', 'too-short', 'too-short', 'Password must be'),
        ('valid-name', 'correct horse battery staple', 'different password value', 'Passwords do not match'),
    ],
)
def test_invalid_public_registration(
    client: AppClient,
    username: str,
    password: str,
    password_confirm: str,
    message: str,
) -> None:
    csrf_token = _registration_csrf(client)
    response = client.post(
        '/auth/register',
        data={
            'csrf_token': csrf_token,
            'username': username,
            'password': password,
            'password_confirm': password_confirm,
        },
    )

    assert response.status_code == 400
    assert message in response.text


def test_duplicate_registration_is_case_insensitive(client: AppClient) -> None:
    csrf_token = _registration_csrf(client)
    response = client.post(
        '/auth/register',
        data={
            'csrf_token': csrf_token,
            'username': 'TestUser',
            'password': 'correct horse battery staple',
            'password_confirm': 'correct horse battery staple',
        },
    )

    assert response.status_code == 400
    assert 'already registered' in response.text


def test_registration_requires_csrf_token(client: AppClient) -> None:
    _registration_csrf(client)
    response = client.post(
        '/auth/register',
        data={
            'username': 'new-user',
            'password': 'correct horse battery staple',
            'password_confirm': 'correct horse battery staple',
        },
    )
    assert response.status_code == 400


def test_registration_can_be_disabled(app: Flask, client: AppClient) -> None:
    app.config['ALLOW_LOCAL_REGISTRATION'] = False
    response = client.get('/auth/register')
    assert response.status_code == 404


@dataclass
class LoginResult:
    target: str   # target URL of the redirect
    content: str  # message/text expected in resulting page
    is_authed: bool = False  # is the result a successful login
    is_admin: bool = False   # is the resulting login an admin user

invalid_login_result = LoginResult(target="/auth/login", content="Invalid username or password.", is_authed=False, is_admin=False)


def check_login(
        client: AppClient,
        username: str,
        password: str,
        next_url: str | None = None,
        *,  # keyword-only beyond here
        expect: LoginResult,
    ) -> None:
    with client:  # so we can use session in get_auth()
        response = client.login(username, password, next_url)

        # We expect a redirect
        assert response.status_code == 302

        # Verify the redirect target
        target = response.headers['Location']
        assert target == expect.target
        response = client.get(target)
        assert response.status_code == 200

        sessauth = get_auth()
        if expect.is_authed:
            # Verify session auth contains correct values for logged-in user
            assert sessauth.user
            assert sessauth.user_id
            assert sessauth.user.display_name == username
            assert sessauth.user.auth_provider == 'local'
            assert sessauth.is_admin == expect.is_admin
            assert sessauth.cur_class is None
        else:
            # Verify session auth contains correct values for non-logged-in user
            assert sessauth.user is None
            assert sessauth.is_admin is False
            assert sessauth.cur_class is None

        # Verify page contents
        assert expect.content in response.text


def test_newuser_command(app: Flask, runner: FlaskCliRunner, client: AppClient) -> None:
    username = "_newuser_"
    check_login(client, username, 'x', expect=invalid_login_result)
    client.logout()

    with app.app_context():
        cmd_result = runner.invoke(args=['newuser', username])
        password_match = re.search(r'password: (\w+)\b', cmd_result.output)
        assert password_match
        password = password_match.group(1)

    with app.test_request_context():
        redir_url = url_for(app.config["DEFAULT_LOGIN_ENDPOINT"])

    check_login(client, username, password, expect=LoginResult(target=redir_url, content="_newuser_", is_authed=True))
    client.logout()
    check_login(client, 'x', password, expect=invalid_login_result)
    client.logout()
    check_login(client, username, 'x', expect=invalid_login_result)


@pytest.mark.parametrize(('username', 'password'), [
    ('', ''),
    ('x', ''),
    ('', 'y'),
    ('x', 'y'),
    ('testuser', 'y'),
    ('testadmin', 'y'),
])
def test_invalid_login(client: AppClient, username: str, password: str) -> None:
    check_login(client, username, password, expect=invalid_login_result)


@pytest.mark.parametrize(('username', 'password', 'next_url', 'is_admin'), [
    ('testuser', 'testpassword', '/profile/', False),
    ('testadmin', 'testadminpassword', '/admin/', True),
])
def test_valid_login(
        app: Flask,
        client: AppClient,
        username: str,
        password: str,
        next_url: str,
        is_admin: bool,
    ) -> None:
    with app.test_request_context():
        redir_url = url_for(app.config["DEFAULT_LOGIN_ENDPOINT"])

    # Test with the next URL specified
    check_login(
        client, username, password, next_url=next_url,
        expect=LoginResult(target=next_url, content=username, is_authed=True, is_admin=is_admin)
    )
    client.logout()
    # Test with no next URL specified: should redirect to /help
    check_login(
        client, username, password, next_url=None,
        expect=LoginResult(target=redir_url, content=username, is_authed=True, is_admin=is_admin)
    )
    client.logout()
    # Test with an unsafe next URL specified: should redirect to /help
    check_login(
        client, username, password, next_url="https://malicious.site/",
        expect=LoginResult(target=redir_url, content=username, is_authed=True, is_admin=is_admin)
    )
    client.logout()


def test_logout(client: AppClient) -> None:
    with client:
        client.login()  # defaults to testuser (id 11)
        sessauth = get_auth()
        assert sessauth.user
        assert sessauth.user.display_name == 'testuser'

        response = client.logout()
        assert response.status_code == 302
        assert response.location == "/auth/login"

        sessauth = get_auth()
        assert sessauth.user is None
        assert sessauth.is_admin is False
        assert sessauth.cur_class is None

        # Check if the user can access the login page and see the flashed message after logout
        response = client.get(response.location)
        assert response.status_code == 200
        assert "You have been logged out." in response.text


@pytest.mark.parametrize(('path', 'nologin', 'withlogin', 'withadmin'), [
    ('/', 200, 200, 200),
    ('/profile/', 302, (200, "2 in the past week"), (200, "Your Profile")),
    ('/help/', 302, 200, 200),
    ('/help/view/1', 302, (400, "Invalid id."), (200, "response01")),
    ('/help/view/999', 302, (400, "Invalid id."), (400, "Invalid id.")),
    ('/tutor/new', 302, 200, 200),
    ('/tutor/1', 302, (200, "user_msg_1"), (200, "user_msg_1")),
    ('/tutor/2', 302, (200, "user_msg_2"), (200, "user_msg_2")),
    ('/tutor/3', 302, (400, "Invalid id."), (200, "user_msg_3")),
    ('/tutor/999', 302, (400, "Invalid id."), (400, "Invalid id.")),
    ('/admin/', 302, (403, "Access denied."), 200),          # admin_required gives 403 Forbidden for non-admin user
    ('/admin/get_db/', 302, (403, "Access denied."), 200),   # admin_required gives 403 Forbidden for non-admin user
])
def test_auth_required(
        client: AppClient,
        path: str,
        nologin: int | tuple[int, str],
        withlogin: int | tuple[int, str],
        withadmin: int | tuple[int, str],
    ) -> None:

    def check_response(response: TestResponse, expected: int | tuple[int, str]) -> None:
        if isinstance(expected, tuple):
            assert response.status_code == expected[0]
            assert expected[1] in response.text
        else:
            assert response.status_code == expected
            if expected == 302:
                assert response.location.startswith('/auth/login')

    response = client.get(path)
    check_response(response, nologin)

    client.login()  # defaults to testuser (id 11)
    client.get('/classes/switch/2')  # switch to class 2 (where the chats are registered)

    response = client.get(path)
    check_response(response, withlogin)

    client.logout()
    response = client.get(path)
    check_response(response, nologin)

    client.login('testadmin', 'testadminpassword')
    response = client.get(path)
    check_response(response, withadmin)

    client.logout()
    response = client.get(path)
    check_response(response, nologin)
