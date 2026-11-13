To generate the meshes:

1. Create a OpenFOAM case with a 0 , constant and system folders -> Use an existing one of the LRN_OGV.
2. Copy run_mesh.sh and the .geo file of one of the 5 meshes found in the folder geo_files
3. Change the name of the .geo file inside run_mesh.sh if needed.
4. Run run_mesh.sh. You will need to have gmsh installed, the version I used is 4.12.1.
5. You can quickly check that everything went okay by looking at the coordinate files in 0.
