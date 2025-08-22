CC      = gcc
CFLAGS  = -Wall -Wextra -Wpedantic -O2 -std=c11

INCLUDE_DIR = include
SRC_DIR     = src
BUILD_DIR   = build
TEST_DIR = test

# -------------------------------------------------------------
# Sources
# -------------------------------------------------------------
LIB_SRCS  := $(wildcard $(SRC_DIR)/*.c)
LIB_OBJS  := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(LIB_SRCS))

TEST_SRCS := $(wildcard $(TEST_DIR)/*.c)
TEST_OBJS := $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_SRCS))

# -------------------------------------------------------------
# Targets
# -------------------------------------------------------------
.PHONY: all clean

all: $(BUILD_DIR)/demo

$(BUILD_DIR)/demo: $(LIB_OBJS) $(TEST_OBJS)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
