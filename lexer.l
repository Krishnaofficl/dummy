%{
#include "y.tab.h"
#include <stdio.h>
#include <stdlib.h>
%}

%%
[0-9]+  { yylval = atoi(yytext); return NUMBER; }
"+"     { return ADD; }
"-"     { return SUB; }
"*"     { return MUL; }
"/"     { return DIV; }
"("     { return '('; }
")"     { return ')'; }
[ \t]+  ;
\n      { return '\n'; }
.       { return yytext[0]; }
%%

int yywrap(void)
{
    return 1;
}
