library(tidyverse)
library(xtable)
library(scales)
library(ggplot2)
library(RColorBrewer)
library(ggpattern)

d <- read_csv("out_comm_u32.csv")
d$total_size <- d$input_size + d$prep_size + d$out_size
d <- d/1000000
d$num_slices <- d$num_slices*1000000
print(d)
ggplot(d, aes(x=num_slices)) +
    labs(
         x = "Num Slices",
         y = "Size (MB)"
    ) +
    theme_minimal() +
    #theme(legend.position = c(0, 1),legend.justification = c(0, 1))+
    geom_line(aes(y=input_size, color = "Input Share")) +
    geom_point(aes(y=input_size, color = "Input Share")) +
    geom_line(aes(y=prep_size, color = "Preparation Share")) +
    geom_point(aes(y=prep_size, color = "Preparation Share")) +
    geom_line(aes(y=out_size, color = "Output Share")) +
    geom_point(aes(y=out_size, color = "Output Share")) +
    geom_line(aes(y=total_size, color = "Total")) +
    geom_point(aes(y=total_size, color = "Total")) +
    scale_colour_manual("",
                      breaks = c("Total","Input Share", "Preparation Share", "Output Share"),
                      values = c("black", "red", "green", "blue")) +
    scale_fill_brewer(palette="Dark2")
ggsave("comm_size.pdf", width = 7, height = 4)
