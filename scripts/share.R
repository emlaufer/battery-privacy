library(tidyverse)
library(xtable)
library(scales)
library(ggplot2)
library(RColorBrewer)
library(ggpattern)

d <- read_csv("out_input_shares_u32.csv")
ggplot(d, aes(x=x, y=size, fill=from)) +
    labs(
         x = "",
         y = "Size (Bytes)"
    ) +
    theme_minimal() +
    theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
    geom_bar(position="stack", stat="identity") +
    scale_fill_brewer(palette="Dark2")
ggsave("input_shares.pdf", width = 4, height = 4)
