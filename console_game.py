from game_2048.logic import Action, Game, game_step
'''Play the game in the console'''


def main():
    game = Game((2, 2))
    while not game.finished:
        print(game)
        try:
            action = input('Enter action: LEFT, RIGHT, UP, DOWN, EXIT\n')
            if action == "EXIT":
                return
            game = game_step(game, Action[action])
        except:
            print('Error: Invalid action')
    print(f'Final score: {game.score}')


if __name__ == '__main__':
    main()
