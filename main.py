import Model.EmotionWave as emw


def main():
    module = emw.EmotionWave(12, 512, 8, 2048,
                    128, 512, 10000, 12, 512,
                    8, 2048)
    print(module)
    print("DONE")

main()