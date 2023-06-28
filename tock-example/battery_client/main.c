#include <led.h>
#include <rng.h>
#include <stdio.h>
#include <stdlib.h>
#include <timer.h>
extern bool field_test();

static int r9_storage;
static int r9_value;

// helpers to expose tock syscalls to rust
void print_rs(char *str, int len) {
  __asm__ volatile("str r9, [%0]\n"
                   "ldr r9, [%1]\n"
                   :
                   : "r"(&r9_storage), "r"(&r9_value));

  printf("%.*s", len, str);
  __asm__ volatile("ldr r9, [%0]\n" : : "r"(&r9_storage));
}

int rand_rs(uint8_t *buf, size_t len) {
  __asm__ volatile("str r9, [%0]\n"
                   "ldr r9, [%1]\n"
                   :
                   : "r"(&r9_storage), "r"(&r9_value));

  int count;
  rng_sync(buf, len, len, &count);

  __asm__ volatile("ldr r9, [%0]\n" : : "r"(&r9_storage));

  return count;
}

void *malloc_rs(size_t bytes) {
  __asm__ volatile("str r9, [%0]\n"
                   "ldr r9, [%1]\n"
                   :
                   : "r"(&r9_storage), "r"(&r9_value));

  void *ret = malloc(bytes);

  __asm__ volatile("ldr r9, [%0]\n" : : "r"(&r9_storage));

  return ret;
}

void free_rs(void *ptr) {
  __asm__ volatile("str r9, [%0]\n"
                   "ldr r9, [%1]\n"
                   :
                   : "r"(&r9_storage), "r"(&r9_value));

  free(ptr);

  __asm__ volatile("ldr r9, [%0]\n" : : "r"(&r9_storage));
}

int main(void) {
  // r9 gets clobbered by rust... save it so we can restore when
  // performing a syscall
  __asm__ volatile("str r9, [%0]\n" : : "r"(&r9_value));

  // Ask the kernel how many LEDs are on this board.
  int num_leds;
  int err = led_count(&num_leds);
  if (err < 0)
    return err;

  printf("Running the battery client...\n");

  // volatile char* killer = 0xDEADBEEF;
  // volatile char my_char = *killer;
  // printf("done\n");
  bool test = run_the_client();
  printf("DONE\n");

  // Blink the LEDs in a binary count pattern and scale
  // to the number of LEDs on the board.
  for (int count = 0;; count++) {
    for (int i = 0; i < num_leds; i++) {
      if (count & (1 << i)) {
        led_on(i);
      } else {
        led_off(i);
      }
    }

    // This delay uses an underlying timer in the kernel.
    delay_ms(500);
  }
}
