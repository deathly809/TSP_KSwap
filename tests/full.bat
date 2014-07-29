@echo off

::For threads 32 through 1024, increment by 32
FOR /L %%T in (32 32 1024) DO (
	::For tilesizes 32 through 1024, increment by 32
	FOR /L %%B in (32 32 1024) DO (
		..\bin\Refactor.exe ..\problems\fl1400.tsp 1000 %%T %%B > output\output_%%T_%%B.txt
	)
)