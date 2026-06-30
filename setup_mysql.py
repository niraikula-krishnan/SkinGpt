"""Create skingpt database and user_analytics table. Run after setting MYSQL_PASSWORD in .env"""

import os
import sys

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

import db

if __name__ == '__main__':
    pwd = os.environ.get('MYSQL_PASSWORD', '')
    if pwd == '':
        print('ERROR: Set MYSQL_PASSWORD in .env first (your MySQL root password).')
        print('Example in .env:')
        print('  MYSQL_PASSWORD=your_mysql_password')
        sys.exit(1)

    if db.init_db():
        print('SUCCESS: Database "skingpt" and table "user_analytics" are ready.')
        for u in db.list_users():
            print(f"  - {u['user_number']}: {u['user_label']} -> {u['analytics']}")
    else:
        print('FAILED: Could not connect to MySQL. Check MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD in .env')
        sys.exit(1)
