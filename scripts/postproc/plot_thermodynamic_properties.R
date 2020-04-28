#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

# SETTINGS =====================================================================

# load libraries
library(argparse)
library(ggplot2)
library(gridExtra)
library(egg)
library(jsonlite)
library(RColorBrewer)


# HANDLE COMMAND LINE ARGUMENTS ================================================

parser <- ArgumentParser(
    description="Plots thermodynamic properties of MD system over time."
)

parser$add_argument(
    "-csv",
    type = "character",
    nargs="?",
    default = "production.csv",
    help = "Name of the OpenMM state data CSV file."
)

args <- parser$parse_args()


# DATA READ-IN =================================================================

# load data
dat <-
    read.table(args$csv, header = TRUE, na.strings = "--", sep = "\t")


# PLOT THEME ===================================================================

# ggplot theme
theme_pub <-
    theme(
        strip.text.x = element_text(
            size = 12,
            margin = margin(0.1, 0.1, 0.1, 0.1, "cm")
        ),
        strip.text.y = element_text(
            size = 12,
            margin = margin(0.1, 0.1, 0.1, 0.1, "cm")
        ),
        text = element_text(size = 13),
        plot.title = element_text(size = 13),
        axis.text.x = element_text(size = 12, colour = "black"),
        axis.text.y = element_text(size = 12, colour = "black"),
        panel.border = element_rect(colour = "black", fill = NA, size = 1),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank()
    )

# dimensions of an A4 page in cm
save_width <- 29.7
save_height <- 21


# PLOTTING =====================================================================

# Temperature ------------------------------------------------------------------

plt_temperature <-
    ggplot(
        dat,
        aes(
            x = Time..ps./1000,
            y = Temperature..K.
        )
    ) +
    geom_line() +
    ggtitle("Temperature") +
    xlab(expression(t ~~"(" * ns * ")")) +
    ylab(expression(T ~~"(" * K * ")")) +
    theme_pub


# Potential Energy -------------------------------------------------------------

plt_potential_energy <-
    ggplot(
        dat,
        aes(
            x = Time..ps./1000,
            y = Potential.Energy..kJ.mole./1000
        )
    ) +
    geom_line() +
    ggtitle("Potential Energy") +
    xlab(expression(t ~~"(" * ns * ")")) +
    ylab(expression(E[pot] ~~"(" * MJ/mol * ")")) +
    theme_pub


# Kinetic Energy ---------------------------------------------------------------

plt_kinetic_energy <-
    ggplot(
        dat,
        aes(
            x = Time..ps./1000,
            y = Kinetic.Energy..kJ.mole./1000
        )
    ) +
    geom_line() +
    ggtitle("Kinetic Energy") +
    xlab(expression(t ~~"(" * ns * ")")) +
    ylab(expression(E[kin] ~~"(" * MJ/mol * ")")) +
    theme_pub


# Total Energy -----------------------------------------------------------------

plt_total_energy <-
    ggplot(
        dat,
        aes(
            x = Time..ps./1000,
            y = Total.Energy..kJ.mole./1000
        )
    ) +
    geom_line() +
    ggtitle("Total Energy") +
    xlab(expression(t ~ ~"(" * ns * ")")) +
    ylab(expression(E[tot] ~ ~"(" * MJ/mol * ")")) +
    theme_pub


# Box Volume -------------------------------------------------------------------

plt_volume <-
    ggplot(
        dat,
        aes(
            x = Time..ps./1000,
            y = Box.Volume..nm.3.
        )
    ) +
    geom_line() +
    ggtitle("Box Volume") +
    xlab(expression(t ~~"(" * ns * ")")) +
    ylab(expression(V ~~"(" * nm^3 * ")")) +
    theme_pub


# Simulation Speed -------------------------------------------------------------

plt_speed <-
    ggplot(
        dat,
        aes(
            x = Time..ps./1000,
            y = Speed..ns.day.
        )
    ) +
    geom_line() +
    ggtitle("Simulation Speed") +
    xlab(expression(t~~"("*ns*")")) +
    ylab(expression(speed~~"("*ns/day*")")) +
    theme_pub


# Combine Plots and Save -------------------------------------------------------

plt_combined <-
    ggarrange(
        plt_potential_energy,
        plt_kinetic_energy,
        plt_total_energy,
        plt_temperature,
        plt_volume,
        plt_speed,
        nrow = 2
    )

ggsave(
    "thermodynamic_properties.png",
    plt_combined,
    width = save_width,
    height = save_height,
    scale = 0.9,
    dpi = 300,
    units = "cm"
)
