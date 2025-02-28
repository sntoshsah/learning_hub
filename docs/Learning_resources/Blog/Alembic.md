# Alembic SQLite Tutorial

## 📌 Step 1: Create a Test Project

```bash
mkdir alembic_sqlite_test
cd alembic_sqlite_test
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install alembic sqlalchemy
```

---

## 📌 Step 2: Initialize Alembic

```bash
alembic init alembic
```

This creates:
- `alembic/` (migrations folder)
- `alembic.ini` (config file)

---

## 📌 Step 3: Configure Alembic for SQLite

Edit **`alembic.ini`** and update the database URL:

```ini
sqlalchemy.url = sqlite:///test.db
```

This sets up an SQLite database file named `test.db`.

---

## 📌 Step 4: Create a SQLAlchemy Model

Create a **`models.py`** file in the project root:

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    email = Column(String(100), unique=True)
```

---

## 📌 Step 5: Link Models with Alembic

Edit **`alembic/env.py`** and find this line:

```python
target_metadata = None
```

Replace it with:

```python
from models import Base
target_metadata = Base.metadata
```

---

## 📌 Step 6: Generate and Apply Migrations

### Generate Migration

```bash
alembic revision --autogenerate -m "create users table"
```

This creates a migration file inside `alembic/versions/`.

### Apply Migration

```bash
alembic upgrade head
```

This applies the migration and creates the `users` table in `test.db`.

---

## 📌 Step 7: Modify the Model and Apply Another Migration

Modify **`models.py`** to add a `created_at` column:

```python
from sqlalchemy import DateTime, func

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    email = Column(String(100), unique=True)
    created_at = Column(DateTime, default=func.now())  # New column
```

### Generate Migration

```bash
alembic revision --autogenerate -m "add created_at column"
```

### Apply Migration

```bash
alembic upgrade head
```

---

## 📌 Step 8: Rollback a Migration

To undo the last migration:

```bash
alembic downgrade -1
```

To downgrade to a specific revision:

```bash
alembic downgrade <revision_id>
```

Find `<revision_id>` in the `alembic/versions/` folder.

---

## ✅ Summary of Commands

| Command | Description |
|---------|------------|
| `alembic init alembic` | Initialize Alembic |
| `alembic revision --autogenerate -m "message"` | Generate migration |
| `alembic upgrade head` | Apply migrations |
| `alembic downgrade -1` | Undo the last migration |

This is a **fully working Alembic setup** with SQLite! 🚀