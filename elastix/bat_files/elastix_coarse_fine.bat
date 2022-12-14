@ECHO OFF

ECHO Registration of INHALE 3D image to EXHALE 3D image for all the patients

ECHO Patient copd1
mkdir C:\work\MIRA\elastix\Output\copd1

elastix -f C:\work\MIRA\chest_ct_registration\data\train\copd1\copd1_iBHCT.nii.gz -m C:\work\MIRA\chest_ct_registration\data\train\copd1\copd1_eBHCT.nii.gz -out C:\work\MIRA\elastix\Output\copd1 -p C:\work\MIRA\parameters\Parameters.MI.Coarse.Bspline_tuned.txt -p C:\work\MIRA\parameters\Parameters.MI.Fine.Bspline_tuned.txt

ECHO Patient copd2
mkdir C:\work\MIRA\elastix\Output\copd2

elastix -f C:\work\MIRA\chest_ct_registration\data\train\copd2\copd2_iBHCT.nii.gz -m C:\work\MIRA\chest_ct_registration\data\train\copd2\copd2_eBHCT.nii.gz -out C:\work\MIRA\elastix\Output\copd2 -p C:\work\MIRA\parameters\Parameters.MI.Coarse.Bspline_tuned.txt -p C:\work\MIRA\parameters\Parameters.MI.Fine.Bspline_tuned.txt

ECHO Patient copd3
mkdir C:\work\MIRA\elastix\Output\copd3

elastix -f C:\work\MIRA\chest_ct_registration\data\train\copd3\copd3_iBHCT.nii.gz -m C:\work\MIRA\chest_ct_registration\data\train\copd3\copd3_eBHCT.nii.gz -out C:\work\MIRA\elastix\Output\copd3 -p C:\work\MIRA\parameters\Parameters.MI.Coarse.Bspline_tuned.txt -p C:\work\MIRA\parameters\Parameters.MI.Fine.Bspline_tuned.txt
 
ECHO Patient copd4
mkdir C:\work\MIRA\elastix\Output\copd4

elastix -f C:\work\MIRA\chest_ct_registration\data\train\copd4\copd4_iBHCT.nii.gz -m C:\work\MIRA\chest_ct_registration\data\train\copd4\copd4_eBHCT.nii.gz -out C:\work\MIRA\elastix\Output\copd4 -p C:\work\MIRA\parameters\Parameters.MI.Coarse.Bspline_tuned.txt -p C:\work\MIRA\parameters\Parameters.MI.Fine.Bspline_tuned.txt


PAUSE