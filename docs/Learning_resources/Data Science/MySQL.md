


# MySQL Documentation

Welcome to the MySQL Documentation! This guide is designed to help you master MySQL, including installation, basic commands, advanced queries, and administration tasks.

---

## Getting Started

### Installation

- **Ubuntu:**
  ```bash
  sudo apt update
  sudo apt install mysql-server
  sudo systemctl start mysql
  sudo systemctl enable mysql
  ```

- **macOS (Homebrew):**
  ```bash
  brew install mysql
  brew services start mysql
  ```

- **Windows:**
  - Download the MySQL Installer from the [official website](https://dev.mysql.com/downloads/installer/).
  - Follow the installation wizard and set the root password.

### Verifying Installation
To confirm MySQL is installed:
```bash
mysql --version
```

---

## Basic Commands

### Starting and Stopping MySQL Service
- **Start MySQL:**
  ```bash
  sudo systemctl start mysql
  ```
- **Stop MySQL:**
  ```bash
  sudo systemctl stop mysql
  ```
- **Restart MySQL:**
  ```bash
  sudo systemctl restart mysql
  ```

---

### Connecting to MySQL
To connect as the root user:
```bash
mysql -u root -p
```
- **`-u`:** Specifies the username.
- **`-p`:** Prompts for the password.

---

## Database Operations

### Create a Database
```sql
CREATE DATABASE database_name;
```
- **Example:**
  ```sql
  CREATE DATABASE test_db;
  ```

### Show Databases
```sql
SHOW DATABASES;
```

### Select a Database
```sql
USE database_name;
```

### Drop a Database
```sql
DROP DATABASE database_name;
```

---

## Table Operations

### Create a Table
```sql
CREATE TABLE table_name (
    column1 datatype,
    column2 datatype,
    ...
);
```
- **Example:**
  ```sql
  CREATE TABLE users (
      id INT AUTO_INCREMENT PRIMARY KEY,
      name VARCHAR(50),
      email VARCHAR(100)
  );
  ```

### Show Tables
```sql
SHOW TABLES;
```

### Describe a Table
```sql
DESCRIBE table_name;
```

### Drop a Table
```sql
DROP TABLE table_name;
```

---

## CRUD Operations

### Insert Data
```sql
INSERT INTO table_name (column1, column2, ...) VALUES (value1, value2, ...);
```
- **Example:**
  ```sql
  INSERT INTO users (name, email) VALUES ('John Doe', 'john@example.com');
  ```

### Select Data
```sql
SELECT column1, column2 FROM table_name;
```
- **Example:**
  ```sql
  SELECT * FROM users;
  ```

### Update Data
```sql
UPDATE table_name SET column1 = value1 WHERE condition;
```
- **Example:**
  ```sql
  UPDATE users SET email = 'new_email@example.com' WHERE id = 1;
  ```

### Delete Data
```sql
DELETE FROM table_name WHERE condition;
```
- **Example:**
  ```sql
  DELETE FROM users WHERE id = 1;
  ```

---

## User Management

### Create a User
```sql
CREATE USER 'username'@'localhost' IDENTIFIED BY 'password';
```

### Grant Privileges
```sql
GRANT ALL PRIVILEGES ON database_name.* TO 'username'@'localhost';
```

### Show User Privileges
```sql
SHOW GRANTS FOR 'username'@'localhost';
```

### Revoke Privileges
```sql
REVOKE ALL PRIVILEGES ON database_name.* FROM 'username'@'localhost';
```

### Drop a User
```sql
DROP USER 'username'@'localhost';
```

---

## Backup and Restore

### Backup a Database
```bash
mysqldump -u username -p database_name > backup.sql
```

### Restore a Database
```bash
mysql -u username -p database_name < backup.sql
```

---

## Common Administrative Tasks

### Reset Root Password
1. Stop MySQL:
   ```bash
   sudo systemctl stop mysql
   ```
2. Start MySQL in Safe Mode:
   ```bash
   sudo mysqld_safe --skip-grant-tables &
   ```
3. Connect to MySQL:
   ```bash
   mysql -u root
   ```
4. Reset the Password:
   ```sql
   ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';
   ```
5. Restart MySQL:
   ```bash
   sudo systemctl restart mysql
   ```

### Check MySQL Status
```bash
sudo systemctl status mysql
```

---

## Advanced Queries

### Joins
- **Inner Join:**
  ```sql
  SELECT * FROM table1
  INNER JOIN table2 ON table1.id = table2.table1_id;
  ```

- **Left Join:**
  ```sql
  SELECT * FROM table1
  LEFT JOIN table2 ON table1.id = table2.table1_id;
  ```

- **Right Join:**
  ```sql
  SELECT * FROM table1
  RIGHT JOIN table2 ON table1.id = table2.table1_id;
  ```

---

## Indexing and Optimization

### Create an Index
```sql
CREATE INDEX index_name ON table_name (column_name);
```

### Drop an Index
```sql
DROP INDEX index_name ON table_name;
```

---

## Troubleshooting

### View Error Logs
- On Linux:
  ```bash
  sudo tail -f /var/log/mysql/error.log
  ```

### Check for Running Queries
```sql
SHOW FULL PROCESSLIST;
```

---

## Resources

- [MySQL Official Documentation](https://dev.mysql.com/doc/)
- [MySQL Tutorials](https://www.mysqltutorial.org/)

---

This documentation is structured to provide step-by-step instructions for MySQL usage and administration tasks. For a complete guide, refer to the official documentation.
```