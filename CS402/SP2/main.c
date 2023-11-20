// main.c

#include <stdio.h>
#include <stdlib.h>
#include "employee.h"
#include "readfile.h"
#include <string.h>
#define MAX_EMPLOYEES 1024

Employee employees[MAX_EMPLOYEES];
int numEmployees = 0;

void loadEmployees(const char *filename);
void printMenu();

void loadEmployees(const char *filename) {
    if (open_file(filename) == NULL) {
        fprintf(stderr, "Failed to open file %s\n", filename);
        exit(1);
    }

    while (numEmployees < MAX_EMPLOYEES && read_int(&employees[numEmployees].id) == 0) {
        read_string(employees[numEmployees].firstName, MAXNAME);
        read_string(employees[numEmployees].lastName, MAXNAME);
        read_int(&employees[numEmployees].salary);
        numEmployees++;
    }

    close_file(currentFile);
}

void printMenu() {
    printf("Employee DB Menu:\n");
    printf("----------------------------------\n");
    printf("  (1) Print the Database\n");
    printf("  (2) Lookup by ID\n");
    printf("  (3) Lookup by Last Name\n");
    printf("  (4) Add an Employee\n");
    printf("  (5) Quit\n");
    printf("  (6) Remove an employee\n");
    printf("  (7) Update an employee's information\n");
    printf("  (8) Print the M employees with the highest salaries\n");
    printf("  (9) Find all employees with matching last name\n");
    printf("----------------------------------\n");
}

void printDatabase() {
    printf("NAME                              SALARY       ID\n");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < numEmployees; i++) {
        printf("%-15s %-15s %d \t %d\n", 
               employees[i].firstName, 
               employees[i].lastName, 
               employees[i].salary, 
               employees[i].id);
    }
    printf("---------------------------------------------------------------\n");
    printf(" Number of Employees (%d)\n", numEmployees);
}

void lookupByID() {
    int id;
    printf("Enter a 6 digit employee id: ");
    scanf("%d", &id);

    for (int i = 0; i < numEmployees; i++) {
        if (employees[i].id == id) {
            printf("NAME                              SALARY       ID\n");
            printf("---------------------------------------------------------------\n");
            printf("%-15s %-15s %d \t %d\n", 
                   employees[i].firstName, 
                   employees[i].lastName, 
                   employees[i].salary, 
                   employees[i].id);
            printf("---------------------------------------------------------------\n");
            return;
        }
    }

    printf("Employee with id %d not found in DB\n", id);
}

void lookupByLastName() {
    char lastName[MAXNAME];
    printf("Enter Employee's last name (no extra spaces): ");
    scanf("%63s", lastName);

    for (int i = 0; i < numEmployees; i++) {
        if (strcmp(employees[i].lastName, lastName) == 0) {
            printf("NAME                              SALARY       ID\n");
            printf("---------------------------------------------------------------\n");
            printf("%-15s %-15s %d \t %d\n", 
                   employees[i].firstName, 
                   employees[i].lastName, 
                   employees[i].salary, 
                   employees[i].id);
            printf("---------------------------------------------------------------\n");
            return;
        }
    }

    printf("No employee with last name '%s' found in DB\n", lastName);
}

void addEmployee() {
    if (numEmployees >= MAX_EMPLOYEES) {
        printf("Database is full. Cannot add more employees.\n");
        return;
    }

    Employee newEmployee;
    int confirm;

    printf("Enter the first name of the employee: ");
    scanf("%63s", newEmployee.firstName);
    
    printf("Enter the last name of the employee: ");
    scanf("%63s", newEmployee.lastName);

    do {
        printf("Enter employee's salary (30000 to 150000): ");
        scanf("%d", &newEmployee.salary);
    } while (newEmployee.salary < 30000 || newEmployee.salary > 150000);

    // Assuming ID is auto-generated and unique
    newEmployee.id = employees[numEmployees - 1].id + 1;
    
    printf("do you want to add the following employee to the DB?\n\t%s %s, salary: %d\nEnter 1 for yes, 0 for no: ", 
           newEmployee.firstName, newEmployee.lastName, newEmployee.salary);
    scanf("%d", &confirm);

    if (confirm) {
        employees[numEmployees++] = newEmployee;
        printf("Employee added successfully.\n");
    }
}

void removeEmployee() {
    int id, indexToRemove = -1;
    printf("Enter Employee ID to remove: ");
    scanf("%d", &id);

    // Find the employee
    for (int i = 0; i < numEmployees; i++) {
        if (employees[i].id == id) {
            indexToRemove = i;
            break;
        }
    }

    if (indexToRemove == -1) {
        printf("Employee not found.\n");
        return;
    }

    // Confirm removal
    int confirm;
    printf("Confirm removal of employee ID %d (1 for Yes, 0 for No): ", id);
    scanf("%d", &confirm);

    if (confirm) {
        // Shift elements to fill the gap
        for (int i = indexToRemove; i < numEmployees - 1; i++) {
            employees[i] = employees[i + 1];
        }
        numEmployees--;
        printf("Employee removed.\n");
    } else {
        printf("Removal cancelled.\n");
    }
}

void updateEmployee() {
    int id;
    printf("Enter Employee ID to update: ");
    scanf("%d", &id);

    // Find the employee
    int indexToUpdate = -1;
    for (int i = 0; i < numEmployees; i++) {
        if (employees[i].id == id) {
            indexToUpdate = i;
            break;
        }
    }

    if (indexToUpdate == -1) {
        printf("Employee not found.\n");
        return;
    }

    // Updating fields
    printf("Enter new first name (or '-' to skip): ");
    char newFirstName[MAXNAME];
    scanf("%s", newFirstName);
    if (strcmp(newFirstName, "-") != 0) {
        strcpy(employees[indexToUpdate].firstName, newFirstName);
    }

    // Repeat for last name and salary
    // ...

    printf("Employee updated.\n");
}

void findAllWithLastName() {
    char lastNameToFind[MAXNAME];
    printf("Enter last name to find: ");
    scanf("%63s", lastNameToFind);

    printf("Employees with last name '%s':\n", lastNameToFind);
    printf("NAME                              SALARY       ID\n");
    printf("---------------------------------------------------------------\n");

    int found = 0;
    for (int i = 0; i < numEmployees; i++) {
        if (strcasecmp(employees[i].lastName, lastNameToFind) == 0) {
            printf("%-15s %-15s %d \t %d\n", 
                   employees[i].firstName, 
                   employees[i].lastName, 
                   employees[i].salary, 
                   employees[i].id);
            found = 1;
        }
    }

    if (!found) {
        printf("No employees found with last name '%s'.\n", lastNameToFind);
    }

    printf("---------------------------------------------------------------\n");
}

void printTopSalaries() {
    int M;
    printf("Enter the number of employees to display: ");
    scanf("%d", &M);

    if (M > numEmployees) M = numEmployees;

    // Temporary array to store indices of top M salaries
    int topIndices[M];

    // Initialize with the first M employees
    for (int i = 0; i < M; i++) {
        topIndices[i] = i;
    }

    // Go through the rest of the employees
    for (int i = M; i < numEmployees; i++) {
        // Find the employee with the lowest salary in the topIndices
        int minIndex = 0;
        for (int j = 1; j < M; j++) {
            if (employees[topIndices[j]].salary < employees[topIndices[minIndex]].salary) {
                minIndex = j;
            }
        }

        // If current employee's salary is higher than the lowest in the topIndices, replace it
        if (employees[i].salary > employees[topIndices[minIndex]].salary) {
            topIndices[minIndex] = i;
        }
    }

    // Print details of top M employees
    printf("Top %d Employees with Highest Salaries:\n", M);
    printf("NAME                              SALARY       ID\n");
    printf("---------------------------------------------------------------\n");
    for (int i = 0; i < M; i++) {
        int idx = topIndices[i];
        printf("%-15s %-15s %d \t %d\n", 
               employees[idx].firstName, 
               employees[idx].lastName, 
               employees[idx].salary, 
               employees[idx].id);
    }
    printf("---------------------------------------------------------------\n");
}

int main(int argc, char *argv[]) {
    // ... [initialization and loading employees]
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    loadEmployees(argv[1]);
    int choice;
    do {
        printMenu();
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                printDatabase();
                break;
            case 2:
                lookupByID();
                break;
            case 3:
                lookupByLastName();
                break;
            case 4:
                addEmployee();
                break;
            case 5:
                printf("goodbye!\n");
                break;
            case 6:
                removeEmployee();
                break;
            case 7:
                updateEmployee();
                break;
            case 8:
                printTopSalaries();
                break;
            case 9:
                findAllWithLastName();
                break;
            default:
                printf("Invalid choice. Please enter a number between 1 and 9.\n");
        }
    } while (choice != 9);

    return 0;
}