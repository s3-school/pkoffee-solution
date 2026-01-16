$out_dir = "build";

# stop compiling on error, by default, show log but do not interact with user
# export INTERACTION=batchmode to disable printing the log to stdout
$lualatex = "lualatex -shell-escape -interaction=\${INTERACTION:-nonstopmode} -halt-on-error %O %S";
# use lualatex
$pdf_mode = 4;

# custom rule to support running makeglossaries as part of latexmk
add_cus_dep('glo', 'gls', 0, 'makeglossaries');
sub makeglossaries {
   # needed to support "out_dir=build", get dirname of the output file
   $dir = dirname($_[0]);
   $file = basename($_[0]);
   system( "makeglossaries", "-d", $dir, $file);
}

push @generated_exts, "bcf bbl acn acr alg ist slg slo sls";
$clean_ext = "bcf bbl glg glo gls acn acr alg ist slg slo sls run.xml xdy snm nav";

# ignore timestamps in minted files for re-running, otherwise results in infinite loop
$hash_calc_ignore_pattern{'minted'}='timestamp';
