ECHO OFF
ECHO Tranformix landmarks

ECHO copd1
mkdir C:\work\MIRA\elastix\Output_transformix\copd1

transformix -def C:\work\MIRA\chest_ct_registration\data\train\copd1\copd1_300_iBH_xyz_r1.txt -out C:\work\MIRA\elastix\Output_transformix\copd1 -tp C:\work\MIRA\elastix\Output_coarse_fine\copd1\TransformParameters.1.txt

ECHO copd2
mkdir C:\work\MIRA\elastix\Output_transformix\copd2

transformix -def C:\work\MIRA\chest_ct_registration\data\train\copd2\copd2_300_iBH_xyz_r1.txt -out C:\work\MIRA\elastix\Output_transformix\copd2 -tp C:\work\MIRA\elastix\Output_coarse_fine\copd2\TransformParameters.1.txt

ECHO copd3
mkdir C:\work\MIRA\elastix\Output_transformix\copd3

transformix -def C:\work\MIRA\chest_ct_registration\data\train\copd3\copd3_300_iBH_xyz_r1.txt -out C:\work\MIRA\elastix\Output_transformix\copd3 -tp C:\work\MIRA\elastix\Output_coarse_fine\copd3\TransformParameters.1.txt

ECHO copd4
mkdir C:\work\MIRA\elastix\Output_transformix\copd4

transformix -def C:\work\MIRA\chest_ct_registration\data\train\copd4\copd4_300_iBH_xyz_r1.txt -out C:\work\MIRA\elastix\Output_transformix\copd4 -tp C:\work\MIRA\elastix\Output_coarse_fine\copd4\TransformParameters.1.txt

ECHO Tranformix terminado
PAUSE
