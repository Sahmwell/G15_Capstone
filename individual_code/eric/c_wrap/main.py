from env.SumoEnv import SumoEnv
import time

def main():
    test = SumoEnv(1000, False)
    test.reset()

if __name__ == '__main__':
    main()
