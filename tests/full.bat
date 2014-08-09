;;@echo on

::For threads 32 through %3, increment by 32
FOR /L %%T in (128 32 %3) DO (
	::For tilesizes 32 through %4, increment by 32
	FOR /L %%B in (32 32 %4) DO (
		%1 ..\problems\fl1400.tsp 1000 %%T %%B > output\%2_output_%%T_%%B.txt
	)
)