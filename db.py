"""MySQL storage for per-user disease analytics. Users are added dynamically per session."""

import json
import os
import uuid
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

_memory: dict[str, dict] = {}
_mysql_ok = False

DEFAULT_USER_NUMBER = 'user_1'
DEFAULT_USER_LABEL = 'User 1'


def _mysql_configured() -> bool:
    return bool(os.environ.get('MYSQL_HOST', '').strip())


@contextmanager
def _connection():
    conn = pymysql.connect(
        host=os.environ.get('MYSQL_HOST', 'localhost'),
        user=os.environ.get('MYSQL_USER', 'root'),
        password=os.environ.get('MYSQL_PASSWORD', ''),
        database=os.environ.get('MYSQL_DATABASE', 'skingpt'),
        charset='utf8mb4',
        cursorclass=DictCursor,
        autocommit=True,
    )
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> bool:
    """Create database and table. Does not seed users — they are added per session."""
    global _mysql_ok
    if not _mysql_configured():
        _mysql_ok = False
        return False
    try:
        db_name = os.environ.get('MYSQL_DATABASE', 'skingpt')
        conn = pymysql.connect(
            host=os.environ.get('MYSQL_HOST', 'localhost'),
            user=os.environ.get('MYSQL_USER', 'root'),
            password=os.environ.get('MYSQL_PASSWORD', ''),
            charset='utf8mb4',
            cursorclass=DictCursor,
            autocommit=True,
        )
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        conn.close()

        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS user_analytics (
                        user_number VARCHAR(32) PRIMARY KEY,
                        user_label VARCHAR(80) NOT NULL,
                        analytics JSON NOT NULL,
                        last_disease VARCHAR(100) DEFAULT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                for stmt in (
                    "ALTER TABLE user_analytics MODIFY user_number VARCHAR(32)",
                    "ALTER TABLE user_analytics MODIFY user_label VARCHAR(80)",
                    "ALTER TABLE user_analytics ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                ):
                    try:
                        cur.execute(stmt)
                    except Exception:
                        pass
        _mysql_ok = True
        return True
    except Exception as exc:
        print(f'MySQL unavailable, using in-memory analytics: {exc}')
        _mysql_ok = False
        return False


def db_active() -> bool:
    return _mysql_ok


def _parse_analytics(raw) -> dict:
    if isinstance(raw, str):
        return json.loads(raw) if raw else {}
    return raw or {}


def _fetch_users() -> list[dict]:
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT user_number, user_label, analytics FROM user_analytics ORDER BY created_at'
                )
                rows = cur.fetchall()
        return [
            {
                'user_number': row['user_number'],
                'user_label': row['user_label'],
                'analytics': _parse_analytics(row['analytics']),
            }
            for row in rows
        ]

    return [
        {
            'user_number': uid,
            'user_label': data['user_label'],
            'analytics': dict(data['analytics']),
        }
        for uid, data in _memory.items()
    ]


def seed_default_user() -> dict:
    """Ensure User 1 exists at the start of each session."""
    if user_exists(DEFAULT_USER_NUMBER):
        return {
            'user_number': DEFAULT_USER_NUMBER,
            'user_label': DEFAULT_USER_LABEL,
            'analytics': get_analytics(DEFAULT_USER_NUMBER),
        }
    row = {
        'user_number': DEFAULT_USER_NUMBER,
        'user_label': DEFAULT_USER_LABEL,
        'analytics': {},
    }
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_analytics (user_number, user_label, analytics)
                    VALUES (%s, %s, %s)
                    """,
                    (DEFAULT_USER_NUMBER, DEFAULT_USER_LABEL, json.dumps({})),
                )
    else:
        _memory[DEFAULT_USER_NUMBER] = {
            'user_label': DEFAULT_USER_LABEL,
            'analytics': {},
            'last_disease': None,
        }
    return row


def list_users() -> list[dict]:
    users = _fetch_users()
    if not users:
        seed_default_user()
        users = _fetch_users()
    return users


def user_exists(user_number: str) -> bool:
    if not user_number:
        return False
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT 1 FROM user_analytics WHERE user_number = %s LIMIT 1',
                    (user_number,),
                )
                return cur.fetchone() is not None
    return user_number in _memory


def get_user_label(user_number: str) -> str:
    if not user_number:
        return 'Unknown'
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT user_label FROM user_analytics WHERE user_number = %s',
                    (user_number,),
                )
                row = cur.fetchone()
        return row['user_label'] if row else user_number
    return _memory.get(user_number, {}).get('user_label', user_number)


def create_user(user_label: str) -> dict:
    label = (user_label or '').strip() or f'User {len(list_users()) + 1}'
    user_number = f'u_{uuid.uuid4().hex[:8]}'
    row = {'user_number': user_number, 'user_label': label, 'analytics': {}}

    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO user_analytics (user_number, user_label, analytics)
                    VALUES (%s, %s, %s)
                    """,
                    (user_number, label, json.dumps({})),
                )
        return row

    _memory[user_number] = {'user_label': label, 'analytics': {}, 'last_disease': None}
    return row


def get_analytics(user_number: str) -> dict[str, int]:
    if not user_exists(user_number):
        return {}
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT analytics FROM user_analytics WHERE user_number = %s',
                    (user_number,),
                )
                row = cur.fetchone()
        return _parse_analytics(row['analytics']) if row else {}

    return dict(_memory[user_number]['analytics'])


def _get_last_disease(user_number: str) -> str | None:
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT last_disease FROM user_analytics WHERE user_number = %s',
                    (user_number,),
                )
                row = cur.fetchone()
        return row['last_disease'] if row else None
    return _memory.get(user_number, {}).get('last_disease')


def _set_user_state(user_number: str, analytics: dict, last_disease: str | None) -> None:
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE user_analytics
                    SET analytics = %s, last_disease = %s
                    WHERE user_number = %s
                    """,
                    (json.dumps(analytics), last_disease, user_number),
                )
        return

    if user_number in _memory:
        _memory[user_number]['analytics'] = dict(analytics)
        _memory[user_number]['last_disease'] = last_disease


def record_disease(user_number: str, disease: str, display_fn) -> dict[str, int]:
    if not user_exists(user_number):
        return {}
    display = display_fn(disease)
    last = _get_last_disease(user_number)
    counts = get_analytics(user_number)

    if last != display:
        counts[display] = counts.get(display, 0) + 1
        _set_user_state(user_number, counts, display)

    return counts


def reset_all_users() -> None:
    """Clear session and restore default User 1 (browser refresh)."""
    global _memory
    _memory = {}
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM user_analytics')
    seed_default_user()
