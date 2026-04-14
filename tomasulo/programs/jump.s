	lw	R1	R0	op1	
	lw	R2	R0	op2	 
	lw	R3	R0	op3	
	add	R4	R1	R2	
loop	sub	R5	R1	R3	
	sw	R5	R0	answer	
	j	loop
done	halt
op1	.dw	50			
op2	.dw	30
op3	.dw	20
answer	.dw	0