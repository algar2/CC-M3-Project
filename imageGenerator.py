"""
Alfonso Garcia
CSCI 3725: Computational Creativity
M3: Markov Distinction
17 September 2024

This file generates an image in the Mintedian symbolic language (my personal art style's name). 

"""

# Importing dependecies

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grids
from PIL import Image 

# Uncomment if deciding to use csv as an input
#import csv



# Initial transition matrix of Mintedian symbols

MATRIX_VALUES = {"eye": {"eye": None, "rocket": None, "watcher": None, "crown": None, "heart": None, "head": None, "tv": None, "flame": None, "mushroom": None, "hand": None, "cell": None, "creator": None},
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

    def __init__(self, transition_matrix=MATRIX_VALUES):
            self.transition_matrix = transition_matrix
            self.symbols = list(transition_matrix.keys())


    def rank_from_input(self):
        """
        Takes in user input of a file name with user's preferences listed in the form "rank_number, symbol_name.
        For each line in the file, strips and splits to organize into a tuple in the format: (rank_number, symbol_name)
        Using this tuple, stores symbol_name as a key ina dictionary user_ranking with rank_number as its value. 
        
        Returns:
            user_ranking: dictionary with symbols mapped to their rankings
        """
        user_ranking = {}
        ranking = 1

        file_to_read = input()
        
        with open(file_to_read, 'r') as file:
            for line in file:

                line = line.strip()

                rank_tuple = tuple(line.split(', '))

                if (rank_tuple[1] in self.symbols):
                    user_ranking[rank_tuple[1]] = rank_tuple[0]
                ranking += 1
        return user_ranking


 
    # def rank_from_csv(self, file_name):
    #     """
    #     Takes in a csv file name and reads it to generate the user_ranking dictionary 

    #     Args:
    #         file_name: file of csv to read
        
    #     Returns:
    #         user_ranking: dictionary with symbols mapped to their rankings
    #     """

    #     symbol_keys = []
    #     symbol_vals = []

    #     user_ranking = {}

    #     with open(file_name, 'r') as file:

    #         csvreader = csv.reader(file)

    #         read_first = 0

    #         for row in csvreader:
    #             if read_first == 1:
    #                 symbol_vals = row
    #             else:
    #                 symbol_keys =row
    #                 read_first += 1
            
    #         file.close()

    #     symbol_keys.pop(0)
    #     symbol_vals.pop(0)

    #     for key, val in zip(symbol_keys, symbol_vals):
    #         user_ranking[key.lower()] = val

    #     return user_ranking

    def get_next_symbol(self, current_symbol):
        """
        Decides which symbol to pick next

        Args: 
            current_symbol - current symbol plotted
        """
        return np.random.choice(
            self.symbols,
            p=[self.transition_matrix[current_symbol][next_symbol] for next_symbol in self.symbols]
        )
    
    def create_canvas(self, current_symbol="eye", sequence_length = 40):
        """
        Creating a list of symbols by iterating through the transition matrix

        Args: 
            current_symbol: what symbol to start with, set to "eye" by default
            sequence_length: how many symbols to plot, set to 40 by default

        Returns:
            canvas: a list with all symbols for Mintedian piece
        """
        canvas = []

        while len(canvas) < sequence_length:
            next_symbol = self.get_next_symbol(current_symbol)
            canvas.append(next_symbol)
            current_symbol = next_symbol
        return canvas
    
    def get_symbol_image(self, current_symbol):
        """
        Helper function for getting the file path of a symbol
        
        Args:
            current_symbol: the symbol whose file path we are looking for
        
        Returns:
            image_path: file path of symbol
        """

        image_path = ""
        match current_symbol:

            case "cell":
                image_path = "assets/cell.png"
            case "creator":
                image_path = "assets/creator.png"
            case "crown":
                image_path = "assets/crown.png"
            case "eye":
                image_path = "assets/eye.png"
            case "flame":
                image_path = "assets/flame.png"
            case "hand":
                image_path = "assets/hand.png"
            case "head":
                image_path = "assets/head.png"
            case "heart":
                image_path = "assets/heart.png"
            case "mushroom":
                image_path = "assets/mushroom.png"
            case "rocket":
                image_path = "assets/rocket.png"
            case "tv":
                image_path = "assets/tv.png"
            case "watcher":
                image_path = "assets/watcher.png"
            case default:
                image_path = ""
        
        return image_path
    
    def get_path_list(self, canvas):
        """
        Stores the file paths of each image within the canvas into a list

        Args:
            canvas: the list with the symbols to plot
        
        Returns:
            file_paths: list of each symbol's filepath
        """
        file_paths = []
        for symbol in canvas:
            file_paths.append(self.get_symbol_image(symbol))
        return file_paths
    
    def random_transformation(self, image):
        """
        Applies a random transformation of a rotation, a vertical flip, or a horizontal flip to an image

        Args: 
            image - an image object 

        Returns:
            image - the transformed image object

        """
        # Possible transformations as a list 
        transformations = ["rotate", "vertical_flip", "horizontal_flip"]

        # Choosing a random transformation
        transformation = np.random.choice(transformations)

        match transformation:
            case "rotate":
                # In the case of rotation, image can be rotated by any amount from 0 to 360 degrees
                image = image.rotate(np.random.randint(0, 360))
            case "vertical_flip":
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            case "horizontal_flip":
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def add_image_to_plot(self, canvas):
        """
        Creates a figure with each move as a subplot by reading in each move's image file_path and plotting it

        Args: 
            canvas - a list of all symbols to plot
        """

        file_paths = self.get_path_list(canvas)

        # Specifying the dimensions of the grid to showcase all 40 symbols

        nrow = 5
        ncol = 8

        # Creating the figure
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

            # Applying a random transformation to an image
            image = self.random_transformation(image)
            ax = plt.subplot(gs[row,col])

            ax.imshow(image)
            # Hiding axis information
            ax.axis('off')

        # Saving Mintedian image to a examples folder: code found https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory
        fig.savefig('examples/example{}.png'.format(5))
        plt.show()
        
    def is_high_low_rank(self, rankings, source_symbol, destination_symbol):
        """
        Boolean function returning whether rank of the source symbol (given that this symbol has already occurred) 
        is greater than the rank of the destination symbol (symbol we want to transition to). 

        Args:
            rankings: dictionary that stores symbol_name as string mapped to its ranking 

        Returns:
            Boolean determining whether source symbol is higher ranked than destination symbol

        """
        return True if rankings[source_symbol] >= rankings[destination_symbol] else False 
    
    def determine_weight(self, source_rank, destination_rank):
        """
        Given a destination and source symbols' ranks, calculate a weight that is proportional to the magnitude of the rank difference.

        If there is a greater difference between the ranks of the symbols (i.e. one is ranked higher on the scale of preference as #2 and the 
        other is ranked lower as #10), weight that is calculated is smaller than if rank difference were to be less (between similarly ranked symbols)
        
        
        given that a desirable, higher ranked symbol has occurred, 
        we want to minimize the chance of transitioning to a lower ranked symbol.

        Args:
            source_rank: the rank of a source symbol, a symbol that has already been plotted
            destination_rank: the rank of a destination symbol, the potential symbol that will be plotted 
        
        Returns:
            A weight proportional to the rank difference 
        """

        rank_diff = abs(int(destination_rank)- int(source_rank))

        # 1 in numerator to generate a weight that when summed over an entire row within transition matrix, weight adds up to 1.
        # Formula found from https://www.indeed.com/career-advice/career-development/normalization-formula
        return 1/(rank_diff + 1)

    def update_transition_matrix(self, rankings):

        """
        Function that updates the probabilities stored in the transition matrix based on user choices. Uses normalization formula
        from https://www.indeed.com/career-advice/career-development/normalization-formula to correctly generate weights (probability
        of transitioning between symbols) of a source symbol (symbol that has just been plotted) to a destiantion symbol (a potential 
        symbol to plot)

        Args:
            rankings: dictionary that stores symbol names mapped to their rankings
        """

        # Creating dictionary for unnormalized weights:
        # Stores as a tuple: (destination_symbol, source_symbol) mapped to the pair's unnormalized probability weight 
         
        unnormalized_weights = {}

        # Storing the total unnormalized probability weight sum of each row to later use for normalization process

        total_sums = []


        # Iterating through transition matrix to calculate unnormalized weights 
        for destination_symbol in self.transition_matrix:
            
            # Getting the sum of raw, unnormalized weights per row
            sum_of_raw_weights = 0

            # Given that a symbol has already been used
            for source_symbol in self.transition_matrix:
                
                # Getting unnormalized weight based on user preference 

                # What should the probability of transitioning from the previous symbol into the next symbol be?
                unnormal_weight = self.determine_weight(rankings[destination_symbol], rankings[source_symbol])

                # Storing unnormalized weight
                unnormalized_weights[(destination_symbol, source_symbol)] = unnormal_weight

                # Getting total sum to use for normalization later 
                sum_of_raw_weights += unnormal_weight

            # Storing sums of each row to a list 
            total_sums.append(sum_of_raw_weights)

        # Now assigning normalized weights based on user preferences, based on unnormalized weights.
        # Step necessary because each row may not sum exactly to 1, so by dividing by the total sum of each row, 
        # a proper probability weight is generated 
             
        for index, destination_rank in enumerate(self.transition_matrix):
            for source_rank in self.transition_matrix:

                # Updating probabilities stored in transition matrix 
                self.transition_matrix[destination_rank][source_rank] = \
               (unnormalized_weights[(destination_rank, source_rank)] / total_sums[index])
                
        
def main():

    # Creating a new ImageGenerator object
    symbol_piece = ImageGenerator()

    # Asking user for input and explaining the symbols within the generator
    print("Hello, welcome to the Mintedian Image Generator" + 
            "\n" + 
            "The following Mintedian symbols are possible to conjure:\n" + 
            "- eye\n" + "-rocket\n" + "-watcher\n" + "-crown\n" + "-heart\n" + "-head\n"
            + "-tv\n" + "-flame\n" + "-mushroom\n" + "-hand\n" + "-cell\n" + "-creator\n"
            "Please rank these from your most favorite to your least favorite and type in a file in the form of rank_number, symbol\n" + 
            "When you have the file, type it in and your Mintedian image will be generated!"
    )

    # Storing the user's input into a dictionary mapping symbols to their respective rank
    rankings = symbol_piece.rank_from_input()

    # Uncomment this code if you want to read input from a csv file
    # rankings = symbol_piece.rank_from_csv("csvs/response_1.csv")


    # Based on user rankings, updating the probabilities stored within the transition matrix
    symbol_piece.update_transition_matrix(rankings)

    # Creating the list of all symbols to plot
    canvas = symbol_piece.create_canvas()

    # Plotting the images and generating the image!
    symbol_piece.add_image_to_plot(canvas)

def __str__(self):
        """Returns a string representation of this ImageGenerator."""
        return self.transition_matrix

def __repr__(self):
        """Lets us make an object of the same value."""
        return "ImageGenerator('{0}', {1})".format(self.transition_matrix, self.symbols)

if __name__ == "__main__": # ask again about this
    main() 

 