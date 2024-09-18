# CC-M3-Project

<b><h2>Mintedian Symbol Generator: </h2></b> <i> Alfonso Garcia</i>

<h3> Description </h3>
A program meant to help guide my creative process for my charcoal drawings. In my work, I often have expanses of shadow where I draw patterns of symbols, as seen below.

<center><img src="images_for_read_me/sample_1.jpg" width="400" height = "800">
<img src="images_for_read_me/sample_2.jpg" width="400" height = "800"></center>

This process is largely intuitive, often done with no set order or path in mind. For this project, I wanted to not only simulate a patch of these symbols through code, but to also help guide my process for future works. I find that having a pattern of symbols to work off of will allow me to create better symbol compositions for future pieces, as in the past when I've created these parts in my works, relying purely on my intuition rendered the process monotonous and boring; I'd just repeat symbols again and again without thinking of what could be more interesting.

Now, before creating these pieces of art, I find it important to brief you on the symbols I'm using, as they're part of both my artistic style and artistic mythology.

The symbols one can rank are as follows:

<center>

<img src ="assets/hand.png" width="100" height="100">
<img src ="assets/cell.png" width="100" height="100">
<img src ="assets/creator.png" width="100" height="100">
<img src ="assets/crown.png" width="100" height="100">
<img src ="assets/eye.png" width="100" height="100">
<img src ="assets/flame.png" width="100" height="100">

<img src ="assets/head.png" width="100" height="100">
<img src ="assets/heart.png" width="100" height="100">
<img src ="assets/mushroom.png" width="100" height="100">
<img src ="assets/rocket.png" width="100" height="100">
<img src ="assets/tv.png" width="100" height="100">
<img src ="assets/watcher.png" width="100" height="100">
</center>
<br>
From left to right and top to bottom, the symbols are: The Hand, Cell, Creator, Crown, Eye, Flame, Head, Heart, Mushroom, Rocket, T.V., and Watcher
<br>
<br>
To run this code, run the imageGenerator.py file, which will await your input for how you rank the symbols in your piece. 
<br>
<h3> How is this meaningful to me? </h3>
As stated above, this sytem is meaningful to me because as an artist, I find relying purely on my intuition to create these patterns in my work hard, especially if I need to cover a lot of areas with it. In this way, having this program can better inform me what symbols I can place next to each other. This program is not meant to create a pattern exactly in my style, but as a starting point that I can then continue to refine when working. 
<br>
<h3>Challenges during M3</h3>
In developing this system, the first big challenge was overcoming the decision paralysis of choosing what to program. All my past experiences in CS have provided me projects with clear expectations and guidance, but this was different because what we chose to do with Markov chains was entirely up to us. From this, I struggled with committing to an idea, let alone one that I found personally meaningful.

However, what I loved about this challenge though was teaching myself through the documentation of matplot and numpy and also looking through StackOverflow and other forums for help, especially with displaying the images. The biggest coding challenge for me was thinking of an algorithm for how to populate my transition matrix with probabilities based on user preference. I looked online on how to normalize and give weights to higher preferred options (thank you indeed.com). The logic of that was what challenged me the most.

Besides this, the project also made me consider how I could directly interact with the user. I made the first attempt of doing this by having an external Google Form that asks users to rank the symbols, then exporting the data as a csv before feeding it into the program. Right now, the default is set to asking the user to rank their choices within the terminal.

This leads to my next step of turning this into a full-stack web application that would ask users for their choices in a pretty interface and then somehow hosting the image generation on the website itself—the nuts and bolts of which I was not able to figure out in a week, but I've started and want to continue implementing it. Reflecting on the process, a lot of it was iterative and I realized how many more features I could add, like image rotation, to make my system more interesting.
<br>

<h3>Is the system creative?</h3>
I think had I not coded this system, I would believe it to be creative. But, understanding how the image it generates was made makes me think my system isn't creative, especially when I can compare its output to what I know I can create in my art pieces. My system is generating probabilities for what symbol occurs and then plotting it on a grid. The assets it uses were drawn by me and its output still lacks the fluidity I'd find in my art. Even if I were to further improve on the system so that it could generate something that looks like my art fully, I still think it would just be a mimic, not something actually creative.
