set output 'time_graph.png'
set xlabel "Threads"
set ylabel "Time (s)"
set title "Time Execution x Thread Numbers"
plot "time.txt" w lines