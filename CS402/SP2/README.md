# Employee Database Program

## Overview
This program is an employee database management system developed in C. It allows users to manage employee records, including adding, updating, removing employees, and querying the database based on various criteria.

## Features
- Load employee data from a file
- Add new employee records
- Remove existing employee records
- Update employee information
- Print the database
- Lookup employees by ID or last name
- Find employees with the highest salaries
- Search for employees with a specific last name

### Prerequisites
- GCC compiler (or any standard C compiler)
- Text editor or IDE of your choice

### Compilation
To compile the program, navigate to the source directory and run the following command:

gcc main.c readfile.c -o employeeDB

Execution
To run the program, use the following command:


./employeeDB input.txt
- Replace input.txt with the path to your input file containing the initial employee data.

## Usage
- After running the program, you will be presented with a menu of options:


1. Print the Database
2. Lookup by ID
3. Lookup by Last Name
4. Add an Employee
5. Quit
6. Remove an employee
7. Update an employee's information
8. Print the M employees with the highest salaries
9. Find all employees with matching last name
10. Select an option by entering the corresponding number.


## Input File Format
- The input file should contain employee data in the following format:

'''php
ID FirstName LastName Salary
- Each field should be separated by a space, and each employee should be on a new line.

## Functions
loadEmployees
Loads employee data from the specified file.

printMenu
Displays the main menu options.

printDatabase
Prints the current state of the employee database.

lookupByID
Searches for an employee by their ID.

lookupByLastName
Finds employees by last name.

addEmployee
Adds a new employee to the database.

removeEmployee
Removes an employee from the database.

updateEmployee
Updates the details of an existing employee.

findAllWithLastName
Finds all employees with a given last name.


## Contributors
[Arman Behnam]
License
You can save this as `README.md` in your project directory. Markdown (.md) is a lightweight markup language with plain-text formatting syntax that is widely used for documentation, especially in projects hosted on platforms like GitHub. Adjust the content as needed to match your project's specifics, such as detailed descriptions of functions, additional setup instructions, or any dependencies.
