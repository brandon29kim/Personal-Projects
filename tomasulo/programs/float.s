	l.s	F0	R0	a	F0 <- a
	l.s	F2	R0	b	F2 <- b
	l.s	F4	R0	c	F4 <- c
	l.s	F8	R0	d
	mult.s	F6	F0	F2	F6 <- a * b
	add.s	F6	F6	F4	F6 <- (a * b) + c
	div.s	F6	F6	F8
	s.s	F6	R0	result	result <- F6
	halt
a	.df	4.5
b	.df	1.5
c	.df	4.003
d	.df	3.23
result	.df	0.0