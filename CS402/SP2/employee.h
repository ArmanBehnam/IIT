// employee.h

#ifndef EMPLOYEE_H
#define EMPLOYEE_H

#define MAXNAME 64

typedef struct {
    int id;
    char firstName[MAXNAME];
    char lastName[MAXNAME];
    int salary;
} Employee;

#endif
