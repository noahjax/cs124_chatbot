# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################
import movielens
import re
import numpy as np


class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
      # The chatbot's default name is `moviebot`. Give your chatbot a new name.
      self.name = 'Lil_Tae'

      self.creative = creative

      # This matrix has the following shape: num_movies x num_users
      # The values stored in each row i and column j is the rating for
      # movie i by user j
      self.titles, ratings = movielens.ratings()
      self.sentiment = movielens.sentiment()

      #############################################################################
      # TODO: Binarize the movie ratings matrix.                                  #
      #############################################################################

      # Binarize the movie ratings before storing the binarized matrix.
      self.ratings = ratings
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
      """Return a message that the chatbot uses to greet the user."""
      #############################################################################
      # TODO: Write a short greeting message                                      #
      #############################################################################

      greeting_message = "How can I help you?"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return greeting_message

    def goodbye(self):
      """Return a message that the chatbot uses to bid farewell to the user."""
      #############################################################################
      # TODO: Write a short farewell message                                      #
      #############################################################################

      goodbye_message = "Have a nice day!"

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return goodbye_message


    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
      """Process a line of input from the REPL and generate a response.

      This is the method that is called by the REPL loop directly with user input.

      You should delegate most of the work of processing the user's input to
      the helper functions you write later in this class.

      Takes the input string from the REPL and call delegated functions that
        1) extract the relevant information, and
        2) transform the information into a response to the user.

      Example:
        resp = chatbot.process('I loved "The Notebok" so much!!')
        print(resp) // prints 'So you loved "The Notebook", huh?'

      :param line: a user-supplied line of text
      :returns: a string containing the chatbot's response to the user input
      """
      #############################################################################
      # TODO: Implement the extraction and transformation in this method,         #
      # possibly calling other functions. Although modular code is not graded,    #
      # it is highly recommended.                                                 #
      #############################################################################
      if self.creative:
        response = "I processed {} in creative mode!!".format(line)
        #print(self.find_movies_by_title('Titanic (1997)'))
        """
        Use this code to call find_movies_closest_to_title
        regex = '"(.+)"'
        titles = re.findall(regex, line)
        self.find_movies_closest_to_title(titles[0])
        """
      else:
        response = "I processed {} in starter mode!!".format(line)

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return response

    def start_word(self, word):
      possible = {'the', 'an', 'a', 'le', 'la'}
      return word.lower() in possible

    def find_match(self, names, text):
      matches = []
      for name in names:

        # format name
        words = name.split(' ')
        if self.start_word(words[-1]):
          words[-2] = words[-2].replace(',', '')
          name = words[-1] + ' ' + ' '.join(word for word in words[:-1])

        # find valid title
        name = name.lower()
        text = text.lower()
        loc  = text.find(name)
        if(loc != -1):
          if not ((loc != 0 and text[loc-1].isalnum()) or (loc+len(name) < len(text) and text[loc+len(name)].isalnum())):
            if not (loc-2 >= 0 and text[loc-2] == 'i' and (loc-3 == -1 or text[loc-3] == ' ')):
              matches.append(name)

      return matches

    def extract_titles(self, text):
      """Extract potential movie titles from a line of text.

      Given an input text, this method should return a list of movie titles
      that are potentially in the text.

      - If there are no movie titles in the text, return an empty list.
      - If there is exactly one movie title in the text, return a list
      containing just that one movie title.
      - If there are multiple movie titles in the text, return a list
      of all movie titles you've extracted from the text.

      Example:
        potential_titles = chatbot.extract_titles('I liked "The Notebook" a lot.')
        print(potential_titles) // prints ["The Notebook"]

      :param text: a user-supplied line of text that may contain movie titles
      :returns: list of movie titles that are potentially in the text
      """
      matches = []

      # add all phrases in quotation marks
      regex = '"(.+?)"'
      prelim = re.findall(regex, text)
      matches.extend(prelim)
      for phrase in prelim: text = text.replace('"' + phrase + '"', '')

      # check for all movies
      for title in self.titles:
        names = []

        # build list of possible titles
        sections = title[0].split(' (')

        # add first and end if only
        names.append(sections[0])
        sections = sections[1:]
        if not sections:
          matches.extend(self.find_match(names, text))
          continue

        # remove date if it exists
        inner = '\d{4}\)'
        if len(re.findall(inner, sections[-1])) != 0: sections = sections[:-1]
        if not sections:
          matches.extend(self.find_match(names, text))
          continue

        # add all alternate titles
        for sect in sections:
          sect = sect.replace(')', '')
          sect = sect.replace('a.k.a. ', '')
          names.append(sect)
          matches.extend(self.find_match(names, text))

      # remove duplicates
      copies = set()
      for i in range(len(matches)):
        for check in matches[i+1:]:
          if matches[i] == check:
            copies.add(i)
      count = 0
      for copy in copies:
        del matches[copy-count]
        count += 1

      return matches

    def find_indices(self, p, query, idx):
      # IMPORTANT: only looks at main, alt, maint+date, and alt+date combinations; not all possible      
      
      # build options
      options = []
      options.append(p['main'])
      if p['date'] != '': options.append(p['main'] + ' (' + p['date'] + ')')
      for alt in p['alt']:
        options.append(alt)
        if p['date'] != '': options.append(alt + ' (' + p['date'] + ')')

      # search for matches
      matches = []
      for opt in options:
        if query.lower() == opt.lower(): matches.append(idx) 
      
      
      return matches

    def find_movies_by_title(self, title):
      """ Given a movie title, return a list of indices of matching movies.

      - If no movies are found that match the given title, return an empty list.
      - If multiple movies are found that match the given title, return a list
      containing all of the indices of these matching movies.
      - If exactly one movie is found that matches the given title, return a list
      that contains the index of that matching movie.

      Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]

      :param title: a string containing a movie title
      :returns: a list of indices of matching movies
      """

      words = title.split(' ')
      if self.start_word(words[0].lower()):
        title = ' '.join(word for word in words[1:]) + ', ' + words[0]

      indices = []
      for i in range(len(self.titles)):
        # build list of possible titles
        title_pieces = {}
        sections = self.titles[i][0].split(' (')

        title_pieces['main'] = sections[0]
        sections = sections[1:]
        if not sections:
          title_pieces['date'] = ''
          title_pieces['alt'] = []
          indices.extend(self.find_indices(title_pieces, title, i))
          continue

        regex = '(\d{4})'
        date = re.findall(regex, sections[-1])
        if date: 
          sections = sections[:-1]
          title_pieces['date'] = date[0]
        if not sections:
          title_pieces['alt'] = []
          indices.extend(self.find_indices(title_pieces, title, i))
          continue

        alts = []
        for sect in sections:
          sect = sect.replace(')', '')
          sect = sect.replace('a.k.a. ', '')
          words = sect.split(' ')
          if self.start_word(words[-1]):
            words[-2] = words[-2].replace(',', '')
            temp = words[-1] + ' ' + ' '.join(word for word in words[:-1])
            alts.append(temp)
        title_pieces['alt'] = alts
        indices.extend(self.find_indices(title_pieces, title, i))
      return indices

    #TODO: Definitely worth implementing +/-2 scoring
    def extract_sentiment(self, text):
      """Extract a sentiment rating from a line of text.

      You should return -1 if the sentiment of the text is negative, 0 if the
      sentiment of the text is neutral (no sentiment detected), or +1 if the
      sentiment of the text is positive.

      As an optional creative extension, return -2 if the sentiment of the text
      is super negative and +2 if the sentiment of the text is super positive.

      Example:
        sentiment = chatbot.extract_sentiment('I liked "The Titanic"')
        print(sentiment) // prints 1

      :param text: a user-supplied line of text
      :returns: a numerical value for the sentiment of the text
      """
      #Create lists of important words for fine grained sentiment
      negations = {"not", "no", "rather", "never", "none", "nobody", "nothing",
                      "neither", "nor", "nowhere", "cannot"}
      strong_words = {'really', 'very', 'especially'}
      strong_pos = {'love', 'ecstatic', 'joy', 'magnificent', 'amazing', 'excellent', 'success'}
      strong_neg = {'hate', 'despise', 'disgust', 'terrible', 'failure', 'disaster'}      

      #Keep track of a sentiment score
      score = 0
      sentiments = movielens.sentiment()
      val = 1
      for word in text.split(' '):
        if word in negations or word.endswith("n't"): val = -1
        if "," in word or "." in word: val = 1
        if word in sentiments:
          if word in strong_words: val *= 2
          if word in strong_pos: score += 2
          elif word in strong_neg: score -= 2
          elif sentiments[word] == 'pos': score += val
          elif sentiments[word] == 'neg': score -= val
      
      if score >= 2: return 2
      elif score > 0: return 1
      elif score == 0: return 0
      elif score > -2: return -1
      else: return -2


    def extract_sentiment_for_movies(self, text):
      """Creative Feature: Extracts the sentiments from a line of text
      that may contain multiple movies. Note that the sentiments toward
      the movies may be different.

      You should use the same sentiment values as extract_sentiment, described above.
      Hint: feel free to call previously defined functions to implement this.

      Example:
        sentiments = chatbot.extract_sentiment_for_text('I liked both "Titanic (1997)" and "Ex Machina".')
        print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

      :param text: a user-supplied line of text
      :returns: a list of tuples, where the first item in the tuple is a movie title,
        and the second is the sentiment in the text toward that movie
      """

    def min_edit(self, source, target):
    
      # Create constants
      n = len(source)+1
      m = len(target)+1
      D = np.zeros((n,m), 'int')
      
      # Init
      for j in range(m):
          D[0][j] = j
      for i in range(n):
          D[i][0] = i
          
      # Recurrence
      for i in range(1,n):
          min_num = float('inf')
          for j in range(1,m):
              if source[i-1] != target[j-1]:
                  D[i][j] = np.min([D[i-1][j]+1, D[i-1][j-1]+1, D[i][j-1]+1])
              else:
                  D[i][j] = np.min([D[i-1][j], D[i-1][j-1], D[i][j-1]]) 
              if D[i][j] < min_num:
                min_num = D[i][j]
          if min_num > 3:
            return 4    
      
      return D[n-1][m-1]

    def find_movies_closest_to_title(self, title, max_distance=3):
      """Creative Feature: Given a potentially misspelled movie title,
      return a list of the movies in the dataset whose titles have the least edit distance
      from the provided title, and with edit distance at most max_distance.

      - If no movies have titles within max_distance of the provided title, return an empty list.
      - Otherwise, if there's a movie closer in edit distance to the given title 
        than all other movies, return a 1-element list containing its index.
      - If there is a tie for closest movie, return a list with the indices of all movies
        tying for minimum edit distance to the given movie.

      Example:
        chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

      :param title: a potentially misspelled title
      :param max_distance: the maximum edit distance to search for
      :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
      """
      # IMPORTANT: only looks for matches to the main title and not alternates or dates
      min_dist = float('inf')
      options = []
      for i in range(len(self.titles)):

        # extract title
        idk = ''
        if '(' not in self.titles[i][0]:
          idk = self.titles[i][0]
        else:
          regex = '(.+) \('
          matches = re.findall(regex, self.titles[i][0])
          idk = matches[0]
        #print(idk)

        # find edit distance
        dist = self.min_edit(title, idk)
        if dist <= max_distance and dist < min_dist:
          min_dist = dist
          options = [i]
        elif dist <= max_distance and dist == min_dist:
          options.append(i)

      return options

    def disambiguate(self, clarification, candidates):
      """Creative Feature: Given a list of movies that the user could be talking about 
      (represented as indices), and a string given by the user as clarification 
      (eg. in response to your bot saying "Which movie did you mean: Titanic (1953) 
      or Titanic (1997)?"), use the clarification to narrow down the list and return 
      a smaller list of candidates (hopefully just 1!)

      - If the clarification uniquely identifies one of the movies, this should return a 1-element
      list with the index of that movie.
      - If it's unclear which movie the user means by the clarification, it should return a list
      with the indices it could be referring to (to continue the disambiguation dialogue).

      Example:
        chatbot.disambiguate("1997", [1359, 2716]) should return [1359]
      
      :param clarification: user input intended to disambiguate between the given movies
      :param candidates: a list of movie indices
      :returns: a list of indices corresponding to the movies identified by the clarification
      """
      all_movies = movielens.titles()
      ans = []
      for index in candidates:
        title_info = all_movies[index]
        title = title_info[0]
        genre = title_info[1]
        if clarification in title or clarification in genre:
          ans.append(index)
      
      return ans


    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    def binarize(self, ratings, threshold=2.5):
      """Return a binarized version of the given matrix.

      To binarize a matrix, replace all entries above the threshold with 1.
      and replace all entries at or below the threshold with a -1.

      Entries whose values are 0 represent null values and should remain at 0.

      :param x: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
      :param threshold: Numerical rating above which ratings are considered positive

      :returns: a binarized version of the movie-rating matrix
      """
      #############################################################################
      # TODO: Binarize the supplied ratings matrix.                               #
      #############################################################################

      # The starter code returns a new matrix shaped like ratings but full of zeros.
      binarized_ratings = np.where(ratings > threshold, 1., -1.)
      zero_mask = np.where(ratings != 0, 1, 0)
      binarized_ratings = np.multiply(binarized_ratings, zero_mask)

      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return binarized_ratings


    def similarity(self, u, v):
      """Calculate the cosine similarity between two vectors.

      You may assume that the two arguments have the same shape.

      :param u: one vector, as a 1D numpy array
      :param v: another vector, as a 1D numpy array

      :returns: the cosine similarity between the two vectors
      """
      #############################################################################
      # TODO: Compute cosine similarity between the two vectors.
      #############################################################################
      similarity = np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return similarity


    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
      """Generate a list of indices of movies to recommend using collaborative filtering.

      You should return a collection of `k` indices of movies recommendations.

      As a precondition, user_ratings and ratings_matrix are both binarized.

      Remember to exclude movies the user has already rated!

      :param user_ratings: a binarized 1D numpy array of the user's movie ratings
      :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
        `ratings_matrix[i, j]` is the rating for movie i by user j
      :param k: the number of recommendations to generate
      :param creative: whether the chatbot is in creative mode

      :returns: a list of k movie indices corresponding to movies in ratings_matrix,
        in descending order of recommendation
      """

      #######################################################################################
      # TODO: Implement a recommendation function that takes a vector user_ratings          #
      # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
      #                                                                                     #
      # For starter mode, you should use item-item collaborative filtering                  #
      # with cosine similarity, no mean-centering, and no normalization of scores.  
      # 
      # Rows are movies, cols are users
      #######################################################################################

      # Populate this list with k movie indices to recommend to the user.

      #Find item similarities
      m,n = ratings_matrix.shape
      item_sims = np.dot(ratings_matrix, ratings_matrix.T)
      item_sims = item_sims.astype(float)
      norms = np.linalg.norm(ratings_matrix, axis=1)

      for i in range(m):
        item_sims[i,:] = item_sims[i,:] / norms[i]
        item_sims[:,i] = item_sims[:,i] / norms[i]

      #Create ratings by weighting similarities by user ratings
      ratings = np.dot(item_sims, user_ratings)

      #Get rid of ratings for movies that the user has already seen
      for i in range(m):
        if user_ratings[i] != 0:
          ratings[i] = -np.inf

      recommendations = list(np.argsort(ratings)[::-1])
      #############################################################################
      #                             END OF YOUR CODE                              #
      #############################################################################
      return recommendations[:k]

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
      """Return debug information as a string for the line string from the REPL"""
      # Pass the debug information that you may think is important for your
      # evaluators
      # debug_info = 'debug info'
      titles = self.extract_titles(line)
      movies = []
      for title in titles:
        movies.append(self.find_movies_by_title(title))
      return movies


    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
      """Return a string to use as your chatbot's description for the user.

      Consider adding to this description any information about what your chatbot
      can do and how the user can interact with it.
      """
      return """
      Your task is to implement the chatbot as detailed in the PA6 instructions.
      Remember: in the starter mode, movie names will come in quotation marks and
      expressions of sentiment will be simple!
      Write here the description for your own chatbot!
      """


if __name__ == '__main__':
  print('To run your chatbot in an interactive loop from the command line, run:')
  print('    python3 repl.py')
