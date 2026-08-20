def check_health(tello):

    battery = tello.get_battery()
    temperature = tello.get_temperature()

    print("\n===== DRONE HEALTH CHECK =====")
    print(f"Battery: {battery}%")
    print(f"Temperature: {temperature} C")

    if battery < 5:
        print("FAILED: Battery is below 5%.")
        return False

    if temperature > 80:
        print("FAILED: Temperature is too high.")
        return False

    print("PASSED: Drone health is OK.")
    return True