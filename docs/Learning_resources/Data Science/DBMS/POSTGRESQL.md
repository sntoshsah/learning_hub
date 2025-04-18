# PostgreSQL: From Installation to Database Management

This guide covers essential PostgreSQL commands and operations for beginners to get started with PostgreSQL on a Unix-based system.

---

## 📦 Installation

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

### Fedora
```bash
sudo dnf install postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl enable postgresql
sudo systemctl start postgresql
```
🧑‍🔧 Managing PostgreSQL Service

### Start PostgreSQL
```bash
sudo systemctl start postgresql
```
### Stop PostgreSQL
```bash
sudo systemctl stop postgresql
```
### Restart PostgreSQL
```bash
sudo systemctl restart postgresql
```
### Check status
```bash
sudo systemctl status postgresql
```
👤 PostgreSQL User and Role Management
### Switch to the postgres user
```bash
sudo -i -u postgres
```
### Access PostgreSQL shell
```bash
psql
```
or 

### Direct Access PostgreSQL shell
```bash
sudo -u postgres psql
```
### Create a new role/user
```sql
CREATE ROLE your_username WITH LOGIN PASSWORD 'your_password';
```
### Grant privileges to the user
```sql
ALTER ROLE your_username CREATEDB;
```

🏗️ Database Operations
### Create a new database
```sql
CREATE DATABASE your_database_name;
```
### Connect to a database
```bash
psql -d your_database_name
```
### Or within psql shell:
```sql
\c your_database_name
```
### List databases
```sql
\l
```
### Drop a database
```sql
DROP DATABASE your_database_name;
```

## 📄 Table Management
### Create a table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50),
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
### Insert data
```sql
INSERT INTO users (username, email) VALUES ('santosh', 'santosh@example.com');
```

### View data
```sql
SELECT * FROM users;
```
### Update data
```sql
UPDATE users SET email = 'new@example.com' WHERE username = 'santosh';
```
### Delete data
```sql
DELETE FROM users WHERE username = 'santosh';
```
🔐 Role and Permission Management
### List roles
```sql
\du
```
### Grant permissions on a database
```sql
GRANT ALL PRIVILEGES ON DATABASE your_database TO your_username;
```
### Revoke permissions
```sql
REVOKE ALL PRIVILEGES ON DATABASE your_database FROM your_username;
```
🛠️ Useful psql Commands
```sql
\q                      -- Quit psql
\dt                     -- List tables
\d table_name           -- Describe table structure
\du                     -- List roles
\l                      -- List databases
\c db_name              -- Connect to database
```
📤 Backup and Restore
### Backup a database
```sql
pg_dump your_database_name > backup.sql
```
### Restore from a backup
```sql
psql your_database_name < backup.sql
```
### Export to CSV
```sql
COPY users TO '/path/to/file.csv' DELIMITER ',' CSV HEADER;
```
### Import from CSV
```sql
COPY users FROM '/path/to/file.csv' DELIMITER ',' CSV HEADER;
```


### 🔄 Reset PostgreSQL Password
```bash
sudo -u postgres psql
```
```sql
ALTER USER postgres WITH PASSWORD 'new_password';
```

🔁 Stored Procedure
### Create Stored Procedure
```sql
CREATE OR REPLACE PROCEDURE increase_salary(emp_id INT, increment NUMERIC)
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE employees
    SET salary = salary + increment
    WHERE id = emp_id;
END;
$$;
```

### Call Stored Procedure
```sql
CALL increase_salary(1, 1000.00);
```

🔔 Trigger
### Step 1: Create Audit Table
```sql
CREATE TABLE employee_audit (
    id SERIAL PRIMARY KEY,
    employee_id INT,
    old_salary NUMERIC,
    new_salary NUMERIC,
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```
### Step 2: Create Trigger Function
```sql
CREATE OR REPLACE FUNCTION log_salary_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.salary <> OLD.salary THEN
        INSERT INTO employee_audit (employee_id, old_salary, new_salary)
        VALUES (OLD.id, OLD.salary, NEW.salary);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```
### Step 3: Create Trigger
```sql
CREATE TRIGGER trigger_salary_change
AFTER UPDATE ON employees
FOR EACH ROW
EXECUTE FUNCTION log_salary_changes();
```
🧪 Verify Trigger
```sql
UPDATE employees
SET salary = salary + 2000
WHERE id = 1;

SELECT * FROM employee_audit;
```

📚 References

[PostgreSQL Documentation](https://www.postgresql.org/docs/current/sql.html)


