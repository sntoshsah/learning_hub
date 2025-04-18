# Database Normalization

Database normalization is the process of organizing the fields and tables of a relational database to minimize redundancy and dependency. It involves dividing large tables into smaller tables and defining relationships between them to enhance data integrity and reduce duplication.

Normalization is carried out in stages, each stage referred to as a "normal form." Below are the key normal forms, their rules, and examples.

---

## Objectives of Normalization

- **Reduce Data Redundancy**: Avoid storing the same data in multiple places.
- **Improve Data Integrity**: Ensure data consistency and correctness.
- **Facilitate Query Optimization**: Enable efficient querying by structuring the data logically.
- **Simplify Maintenance**: Make it easier to update, delete, or add data.

---

## Types of Normal Forms

### 1. First Normal Form (1NF)
A table is in 1NF if:
- Each column contains atomic (indivisible) values.
- Each row is unique.

#### Example (Before 1NF):
| OrderID | Product     | Quantity |
|---------|-------------|----------|
| 1       | Pen, Pencil | 5, 10    |
| 2       | Notebook    | 15       |

#### Problems:
- Non-atomic values (e.g., "Pen, Pencil" in the `Product` column).
- Hard to query specific items.

#### After Applying 1NF:
| OrderID | Product   | Quantity |
|---------|-----------|----------|
| 1       | Pen       | 5        |
| 1       | Pencil    | 10       |
| 2       | Notebook  | 15       |

---

### 2. Second Normal Form (2NF)
A table is in 2NF if:

- It is in 1NF.
- All non-key attributes are fully dependent on the primary key.

#### Example (Before 2NF):
| OrderID | Product   | Quantity | SupplierName |
|---------|-----------|----------|--------------|
| 1       | Pen       | 5        | ABC Supplies |
| 2       | Pencil    | 10       | ABC Supplies |

#### Problems:
- `SupplierName` depends only on `Product`, not on the full composite key (`OrderID, Product`).

#### After Applying 2NF:
**Orders Table:**

| OrderID | Product   | Quantity |
|---------|-----------|----------|
| 1       | Pen       | 5        |
| 2       | Pencil    | 10       |

**Suppliers Table:**

| Product   | SupplierName |
|-----------|--------------|
| Pen       | ABC Supplies |
| Pencil    | ABC Supplies |

---

### 3. Third Normal Form (3NF)
A table is in 3NF if:
- It is in 2NF.
- No transitive dependency exists (non-key attributes do not depend on other non-key attributes).

#### Example (Before 3NF):
| ProductID | ProductName | Category     | CategoryDescription |
|-----------|-------------|--------------|---------------------|
| 101       | Pen         | Stationery   | Writing materials  |
| 102       | Notebook    | Stationery   | Writing materials  |

#### Problems:
- `CategoryDescription` depends on `Category`, not directly on `ProductID`.

#### After Applying 3NF:
**Products Table:**

| ProductID | ProductName | Category   |
|-----------|-------------|------------|
| 101       | Pen         | Stationery |
| 102       | Notebook    | Stationery |

**Categories Table:**

| Category   | CategoryDescription |
|------------|---------------------|
| Stationery | Writing materials   |

---

### 4. Boyce-Codd Normal Form (BCNF)
A table is in BCNF if:

- It is in 3NF.
- Every determinant is a candidate key (a determinant is an attribute that uniquely determines other attributes).

#### Example:
| TeacherID | Course      | Department |
|-----------|-------------|------------|
| T1        | Math        | Science    |
| T2        | Physics     | Science    |
| T3        | Math        | Arts       |

#### Problems:
- `Course` determines `Department`, but `Course` is not a candidate key.

#### After Applying BCNF:
**Teachers Table:**

| TeacherID | Course  |
|-----------|---------|
| T1        | Math    |
| T2        | Physics |
| T3        | Math    |

**Courses Table:**

| Course   | Department |
|----------|------------|
| Math     | Science    |
| Physics  | Science    |
| Math     | Arts       |

---

### 5. Fourth Normal Form (4NF)
A table is in 4NF if:

- It is in BCNF.
- It has no multi-valued dependencies (an attribute depends on another attribute independently of other attributes).

#### Example (Before 4NF):
| StudentID | Course   | Hobby        |
|-----------|----------|--------------|
| 1         | Math     | Reading      |
| 1         | Math     | Painting     |
| 2         | Physics  | Reading      |
| 2         | Physics  | Chess        |

#### Problems:
- `Course` and `Hobby` are independent of each other but are stored together.

#### After Applying 4NF:
**Students-Courses Table:**

| StudentID | Course   |
|-----------|----------|
| 1         | Math     |
| 2         | Physics  |

**Students-Hobbies Table:**

| StudentID | Hobby    |
|-----------|----------|
| 1         | Reading  |
| 1         | Painting |
| 2         | Reading  |
| 2         | Chess    |

---

### 6. Fifth Normal Form (5NF)
A table is in 5NF if:
- It is in 4NF.
- It has no join dependencies (a table should not be decomposable into smaller tables without losing data).

#### Example:
| ProjectID | EmployeeID | Role        |
|-----------|------------|-------------|
| P1        | E1         | Developer   |
| P1        | E2         | Tester      |
| P2        | E1         | Developer   |
| P2        | E3         | Designer    |

#### After Applying 5NF:
**Projects-Employees Table:**

| ProjectID | EmployeeID |
|-----------|------------|
| P1        | E1         |
| P1        | E2         |
| P2        | E1         |
| P2        | E3         |

**Employees-Roles Table:**

| EmployeeID | Role      |
|------------|-----------|
| E1         | Developer |
| E2         | Tester    |
| E3         | Designer  |

---

### 6. Sixth Normal Form (6NF)
A table is in 6NF if:

- It is in 5NF.
- It deals with temporal data and non-additive join dependencies, typically requiring specialized use cases.

#### Example:
Consider a scenario with temporal data where changes in attributes over time are stored, like salary changes for employees.

---

## Benefits of Normalization
1. **Eliminates Redundancy**: Reduces duplication, saving storage space.
2. **Improves Data Consistency**: Updates and deletes are consistent across tables.
3. **Enhances Query Performance**: Structured data allows for faster and more efficient queries.

## When Not to Normalize
Sometimes, denormalization is preferred for:
- Analytical purposes, where joins may hinder performance.
- Situations requiring quick read operations.

---

By understanding and applying normalization, you can design databases that are efficient, maintainable, and reliable for your applications.
