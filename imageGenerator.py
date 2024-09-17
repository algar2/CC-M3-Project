# importing dependecies

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grids
import csv

from PIL import Image 

matrix_values = {"eye": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"rocket": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"watcher": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"crown": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"heart": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"head": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"tv": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"flame": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"mushroom": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"hand": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"cell": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
"creator": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None}}


class imageGenerator: 

    def __init__(self, transition_matrix=matrix_values):
            self.transition_matrix = transition_matrix
            self.symbols = list(transition_matrix.keys())

    # def rank_from_input(self):
    #     """
    #     Takes in user input to assign a ranking based on the order they like 
    #     each symbol from most to least 
    #     """
    #     user_ranking = {}
    #     ranking = 1

    #     file_to_read = input()
        
    #     with open(file_to_read, 'r') as file:
    #         for line in file:

    #             line = line.strip()

    #             rank_tuple = tuple(line.split(', '))

    #             if (rank_tuple[1] in self.symbols):
    #                 user_ranking[rank_tuple[1]] = rank_tuple[0]
    #             ranking += 1

    def rank_from_csv(self, file_name):

        symbol_keys = []
        symbol_vals = []

        symbol_preferences = {}

        with open(file_name, 'r') as file:

            csvreader = csv.reader(file)

            read_first = 0

            for row in csvreader:
                if read_first == 1:
                    symbol_vals = row
                else:
                    symbol_keys =row
                    read_first += 1
            
            file.close()

        symbol_keys.pop(0)
        symbol_vals.pop(0)

        for key, val in zip(symbol_keys, symbol_vals):
            symbol_preferences[key.lower()] = val

        return symbol_preferences


    def get_next_symbol(self, current_symbol):
        """
        Decides which symbol to pick next

        Args: current_symbol (str) - current symbol plotted
        """
        return np.random.choice(
            self.symbols,
            p=[self.transition_matrix[current_symbol][next_symbol] for next_symbol in self.symbols]
        )
    
    def create_canvas(self, current_symbol="eye", sequence_length = 40):

        canvas = []

        while len(canvas) < sequence_length:
            next_symbol = self.get_next_symbol(current_symbol)
            canvas.append(next_symbol)
            current_symbol = next_symbol
        return canvas
    
    def get_symbol_image(self, current_symbol):

        image_path = ""
        match current_symbol:

            case "cell":
                image_path = "symbol_images/cell.png"
            case "creator":
                image_path = "symbol_images/creator.png"
            case "crown":
                image_path = "symbol_images/crown.png"
            case "eye":
                image_path = "symbol_images/eye.png"
            case "flame":
                image_path = "symbol_images/flame.png"
            case "hand":
                image_path = "symbol_images/hand.png"
            case "head":
                image_path = "symbol_images/head.png"
            case "heart":
                image_path = "symbol_images/heart.png"
            case "mushroom":
                image_path = "symbol_images/mushroom.png"
            case "rocket":
                image_path = "symbol_images/rocket.png"
            case "tv":
                image_path = "symbol_images/tv.png"
            case "watcher":
                image_path = "symbol_images/watcher.png"
            case default:
                image_path = ""
        
        return image_path
    
    def get_path_list(self, canvas):
        file_paths = []
        for symbol in canvas:
            file_paths.append(self.get_symbol_image(symbol))
        return file_paths
    
    def random_transformation(self, image):

        transformations = ["rotate", "vertical_flip", "horizontal_flip"]
        transformation = np.random.choice(transformations)

        match transformation:
            case "rotate":
                image = image.rotate(np.random.randint(0, 360))
            case "vertical_flip":
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            case "horizontal_flip":
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return image

    def add_image_to_plot(self, canvas):
        """
        Creates a figure with each move as a subplot by reading in 
        each move's image file_path and plotting it
        """
        file_paths = self.get_path_list(canvas)

        nrow = 5
        ncol = 8

        fig = plt.figure(figsize=(15,8))

        # Got this code from https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots
        # Using GridSpec to create grids flush against each other
        gs = grids.GridSpec(nrow, ncol, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1], 
                            wspace=0.0, hspace=0.0)
                            
        # Setting background color to black
        fig.patch.set_facecolor('black')

        for idx, file_path in enumerate(file_paths):

            # Calculating row and col index based on current index in file_paths
            row = idx // ncol
            col = idx % ncol

            image = Image.open(file_path)
            image = self.random_transformation(image)
            ax = plt.subplot(gs[row,col])

            # Applying a random transformation to an image before sho
            # image = self.random_transformation(image)
            ax.imshow(image)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis('off')

        plt.show()
        
    def is_high_low_rank(self, rankings, key_1, key_2):
        """
        Boolean function returning whether rank of one symbol is higher than rank of 
        other symbol
        """
        return True if rankings[key_1] >= rankings[key_2] else False 
    
    def determine_weight(self, source_rank, destination_rank):
        """
        Function to handle assigining weights of probabilities by order of 
        preference.
        """

        rank_diff = abs(int(destination_rank)- int(source_rank))

        return 1/(rank_diff + 1)

    def update_transition_matrix(self, rankings):

        """
        Function that updates the probabilities stored in the transition matrix
        based on user choices

        Generate probabilities using a random number.
        Iterate through each row
        Calculate whether the row sums to 1 yet
        If not, compare the symbol of the current row's ranking to the 
        symbol of the current column. Then based on this, assign a greater value to
        the higher ranked symbol? then continue to iterate through. 
        """

        # Stores a tuple: (destination_symbol, source_symbol) mapped to the unnormalized weight
         
        unnormalized_weights = {}

        # Storing the total unweighted sum of each row to later use for normalization

        total_sums = []

        for destination_symbol in self.transition_matrix:
            
            # Getting the sum of raw, unnormalized weights per row
            sum_of_raw_weights = 0

            # Given that a symbol has already been used
            for source_symbol in self.transition_matrix:
                
                # Getting unnormalized weight based on user preference 

                unnormal_weight = self.determine_weight(rankings[destination_symbol], rankings[source_symbol])

                unnormalized_weights[(destination_symbol, source_symbol)] = unnormal_weight

                # Getting total sum to use for normalization later 
                sum_of_raw_weights += unnormal_weight

            # Storing sums of each row to a list 
            total_sums.append(sum_of_raw_weights)

        # Now assigning normalized weights based on user preferences, based on unnormalized weights
             
        for index, destination_rank in enumerate(self.transition_matrix):
            for source_rank in self.transition_matrix:

                # Updating probabilities stored in transition matrix 
                self.transition_matrix[destination_rank][source_rank] = \
               (unnormalized_weights[(destination_rank, source_rank)] / total_sums[index])
        
def main():

    symbol_piece = imageGenerator()

    # eye, rocket, watcher, crown , heart, head, tv, flame, mushroom, hand, cell, creator
    print("Hello, welcome to the Mintedian Image Generator" + 
            "\n" + 
            "The following Mintedian symbols are possible to conjure:\n" + 
            "- eye\n" + "-rocket\n" + "-watcher\n" + "-crown\n" + "-heart\n" + "-head\n"
            + "-tv\n" + "-flame\n" + "-mushroom\n" + "-hand\n" + "-cell\n" + "-creator\n"
            "Please rank these from your most favorite to your least favorite in the following form: 1, symbol_x"
    )

    #rankings = symbol_piece.rank_from_input()
    rankings = symbol_piece.rank_from_csv("csvs/response_1.csv")

    symbol_piece.update_transition_matrix(rankings)

    canvas = symbol_piece.create_canvas()

    symbol_piece.add_image_to_plot(canvas)

if __name__ == "__main__": # ask again about this
    main() 

 