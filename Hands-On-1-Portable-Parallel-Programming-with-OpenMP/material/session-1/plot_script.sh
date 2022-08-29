#!/bin/sh
gnuplot
# for j in $(seq 1 10)
# do
#  plot "$j.txt" title "Time execution x Thread numbers", with lines
# done

start = 1
end = 10
files = {$start...$end}
plot for [f in files] f."txt" title f with lines 