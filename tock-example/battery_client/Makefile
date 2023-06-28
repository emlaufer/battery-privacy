# Makefile for user application

# Specify this directory relative to the current application.
TOCK_USERLAND_BASE_DIR = ../..

#TOCK_TARGETS ?= cortex-m4|cortex-m4test|0x00040000|0x20008000

TOCK_TARGETS := cortex-m4|cortex-m4.0x00040048.0x20004000|0x00040048|0x20004000
#TOCK_TARGETS ?= cortex-m4

# Which files to compile.
C_SRCS := $(wildcard *.c)
LIBS_cortex-m4 := librust_app.a
#LAYOUT := layout.ld

STACK_SIZE       ?= 8048
APP_HEAP_SIZE    ?= 32000

# Include userland master makefile. Contains rules and flags for actually
# building the application.
include $(TOCK_USERLAND_BASE_DIR)/AppMakefile.mk

.PHONY: librust_app.a
librust_app.a:
	make -C librust_app build

build: librust_app.a program
