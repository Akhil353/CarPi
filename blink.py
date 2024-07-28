import RPi.GPIO as GPIO

# Set the GPIO mode
GPIO.setmode(GPIO.BOARD)

# Set the GPIO pin number for the LED
LED_PIN = 7

# Set up the LED pin as an output
GPIO.setup(LED_PIN, GPIO.OUT)

# Turn on the LED (set GPIO 17 to HIGH)
GPIO.output(LED_PIN, True)

# Keep the script running indefinitely
try:
    while True:
        pass  # Do nothing, just keep the LED on
except KeyboardInterrupt:
    # Clean up GPIO settings before exiting
    GPIO.cleanup()
