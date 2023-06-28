library(tidyverse)
library(xtable)
library(scales)
library(ggplot2)
library(RColorBrewer)
library(ggpattern)

d <- read_csv("out_times_u32.csv")
ggplot(d, aes(x=num_slices, y=time_s, group=num_clients)) +
    labs(
         x = "Num Slices",
         y = "Time (s)"
    ) +
    theme_minimal() +
    geom_line(aes(color=num_clients)) +
    geom_point(aes(color=num_clients)) +
    scale_fill_brewer(palette="Dark2")
ggsave("time.pdf", width = 7, height = 4)
