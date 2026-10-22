%{
#include <stdio.h>
int yylex(void);
void yyerror(const char *s);
%}

%token NUMBER
%token ADD SUB MUL DIV
%left ADD SUB
%left MUL DIV

%%
input : expr '\n'
      ;

expr  : expr ADD expr
      | expr SUB expr
      | expr MUL expr
      | expr DIV expr
      | '(' expr ')'
      | NUMBER
      ;
%%

int main(void)
{
    int v = yyparse();
    if (v == 0)
        printf("Valid Expression\n");
    return 0;
}

void yyerror(const char *s)
{
    printf("Invalid Expression\n");
}
