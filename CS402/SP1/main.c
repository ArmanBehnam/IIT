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

// ... [previous code segments]

int main(int argc, char *argv[]) {
    // ... [initialization and loading employees]
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    loadEmployees(argv[1]);
    printMenu();
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
            default:
                printf("Invalid choice. Please enter a number between 1 and 5.\n");
        }
    } while (choice != 5);

    return 0;
}