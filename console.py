import logic
'''Play the game in the console'''


def main():
    game = logic.Game((2,2))
    while not game.finished:
        print(game)
        action = input("Enter action: LEFT, RIGHT, UP, DOWN\n")
        game = logic.game_step(game, logic.Action[action])
    print(f"Final score: {game.score}")


if __name__ == "__main__":
    main()
