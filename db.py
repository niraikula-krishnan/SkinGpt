"""MySQL storage for per-user disease analytics (3 users). Falls back to memory if DB unavailable."""

import json
import os
from contextlib import contextmanager

import pymysql
from pymysql.cursors import DictCursor

USERS = [
    ('user_1', 'User 1'),
    ('user_2', 'User 2'),
    ('user_3', 'User 3'),
]

_memory: dict[str, dict] = {
    uid: {'analytics': {}, 'last_disease': None, 'user_label': label}
    for uid, label in USERS
}
_mysql_ok = False


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
    """Create database, table, and seed 3 users. Returns True if MySQL is active."""
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
                        user_number VARCHAR(20) PRIMARY KEY,
                        user_label VARCHAR(50) NOT NULL,
                        analytics JSON NOT NULL,
                        last_disease VARCHAR(100) DEFAULT NULL
                    )
                """)
                for uid, label in USERS:
                    cur.execute(
                        """
                        INSERT INTO user_analytics (user_number, user_label, analytics)
                        VALUES (%s, %s, %s)
                        ON DUPLICATE KEY UPDATE user_label = VALUES(user_label)
                        """,
                        (uid, label, json.dumps({})),
                    )
        _mysql_ok = True
        return True
    except Exception as exc:
        print(f'MySQL unavailable, using in-memory analytics: {exc}')
        _mysql_ok = False
        return False


def db_active() -> bool:
    return _mysql_ok


def list_users() -> list[dict]:
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT user_number, user_label, analytics, last_disease FROM user_analytics ORDER BY user_number'
                )
                rows = cur.fetchall()
        result = []
        for row in rows:
            analytics = row['analytics']
            if isinstance(analytics, str):
                analytics = json.loads(analytics)
            result.append({
                'user_number': row['user_number'],
                'user_label': row['user_label'],
                'analytics': analytics or {},
            })
        return result

    return [
        {
            'user_number': uid,
            'user_label': _memory[uid]['user_label'],
            'analytics': dict(_memory[uid]['analytics']),
        }
        for uid, _ in USERS
    ]


def get_analytics(user_number: str) -> dict[str, int]:
    user_number = _valid_user(user_number)
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    'SELECT analytics FROM user_analytics WHERE user_number = %s',
                    (user_number,),
                )
                row = cur.fetchone()
        if not row:
            return {}
        analytics = row['analytics']
        if isinstance(analytics, str):
            analytics = json.loads(analytics)
        return analytics or {}

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
    return _memory[user_number]['last_disease']


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

    _memory[user_number]['analytics'] = dict(analytics)
    _memory[user_number]['last_disease'] = last_disease


def record_disease(user_number: str, disease: str, display_fn) -> dict[str, int]:
    """Increment disease count only when it differs from the previous disease for this user."""
    user_number = _valid_user(user_number)
    display = display_fn(disease)
    last = _get_last_disease(user_number)
    counts = get_analytics(user_number)

    if last != display:
        counts[display] = counts.get(display, 0) + 1
        _set_user_state(user_number, counts, display)

    return counts


def reset_all_users() -> None:
    """Clear analytics for all users (called on browser session refresh)."""
    if _mysql_ok:
        with _connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE user_analytics SET analytics = %s, last_disease = NULL",
                    (json.dumps({}),),
                )
        return

    for uid in _memory:
        _memory[uid]['analytics'] = {}
        _memory[uid]['last_disease'] = None


def _valid_user(user_number: str) -> str:
    allowed = {uid for uid, _ in USERS}
    if user_number in allowed:
        return user_number
    return 'user_1'
    
