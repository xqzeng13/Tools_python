import imagej
import imagej as IJ
ij = imagej.init()

rm = IJ.RoiManager.getRoiManager();
imp1 = IJ.openImage("C:/Users/hello/Desktop/test/RhoA-c-11.tif")
imp2 = IJ.openImage("C:/Users/hello/Desktop/test/RhoA-c-11.png")
imp2.setAutoThreshold("Default dark no-reset")
#Prefs.blackBackground = true;
IJ.run(imp2, "Convert to Mask", "")
IJ.run(imp2, "Analyze Particles...", "  show=Outlines exclude add composite")
IJ.run("Set Measurements...", "mean modal min integrated display redirect=None decimal=3");
rm.runCommand(imp1,"Show None")
rm.runCommand(imp1,"Show All")
rm.runCommand(imp1,"Measure")
imp1.show()