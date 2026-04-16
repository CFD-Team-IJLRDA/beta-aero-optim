# script.gp

set logscale y
set grid
lw = 2

# --- ADP: shades of blue ---
color_ADP_1 = "#1f77b4"
color_ADP_2 = "#4f9add"
color_ADP_3 = "#85bdf7"

# --- OP1: shades of green ---
color_OP1_1 = "#2ca02c"
color_OP1_2 = "#58c658"
color_OP1_3 = "#8fe88f"

# --- OP2: shades of orange ---
color_OP2_1 = "#ff7f0e"
color_OP2_2 = "#ff9f42"
color_OP2_3 = "#ffbf73"

plot sprintf("%s/ADP/residualRel.dat", dir) using 1:3 with lines lw lw lc rgb color_ADP_1 title "ADP - Residual 3", \
    sprintf("%s/ADP/residualRel.dat", dir) using 1:4 with lines lw lw lc rgb color_ADP_2 title "ADP - Residual 4", \
    sprintf("%s/ADP/residualRel.dat", dir) using 1:5 with lines lw lw lc rgb color_ADP_3 title "ADP - Residual 5", \
    sprintf("%s/OP1/residualRel.dat", dir) using 1:3 with lines lw lw lc rgb color_OP1_1 title "OP1 - Residual 3", \
    sprintf("%s/OP1/residualRel.dat", dir) using 1:4 with lines lw lw lc rgb color_OP1_2 title "OP1 - Residual 4", \
    sprintf("%s/OP1/residualRel.dat", dir) using 1:5 with lines lw lw lc rgb color_OP1_3 title "OP1 - Residual 5", \
    sprintf("%s/OP2/residualRel.dat", dir) using 1:3 with lines lw lw lc rgb color_OP2_1 title "OP2 - Residual 3", \
    sprintf("%s/OP2/residualRel.dat", dir) using 1:4 with lines lw lw lc rgb color_OP2_2 title "OP2 - Residual 4", \
    sprintf("%s/OP2/residualRel.dat", dir) using 1:5 with lines lw lw lc rgb color_OP2_3 title "OP2 - Residual 5"

