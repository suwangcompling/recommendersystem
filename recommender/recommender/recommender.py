from collections import Counter, defaultdict
from functools import partial
from operator import itemgetter
import numpy as np

class Recommender:
    
    def __init__(self, data):
        # data: a list of lists of user interests, each sublist is a user
        # mode: most_popular, user_oriented, interest_oriented.
        self.data = data
        self.modesInfo = defaultdict(list) # one data entry each mode.
            # popularityRanks for 'most_popular'
            # usersInterestsMatrix for 'user_oriented'
            # interestsUsersMatrix for 'interest_oriented'
        self.recommendCount = 0
        self.newData = []
        self.retrain = False
        self.__train()
    
    # TRAINING FUNCTIONS
    
    def __train(self, userInterests=[]): # TODO: later, train all three mode when called!
        self.recommendCount += 1
        print "... training"
        self.__most_pupular_train()
        self.__user_oriented_train()
        self.__item_oriented_train()
        # update database after 10 recommendations.
        self.newData.append(userInterests)
        if self.recommendCount % 10 == 0:
            self.data.extend(self.newData) 
            self.newData = []
            self.retrain = True
    
    def __most_pupular_train(self):
        popularityCounts = Counter(entry for datum in self.data for entry in datum)
            # datum is an array of entries (e.g. user interests).
        popularityRanks = sorted(popularityCounts.items(), key=itemgetter(1), reverse=True)
        self.modesInfo['most_popular'] = popularityRanks
                
    def __user_oriented_train(self):
        self.uniqueInterests = sorted(list({entry for datum in self.data for entry in datum}))
        self.interestToIndex = {interest:i for i,interest in enumerate(self.uniqueInterests)}
        usersInterestsMatrix = map(self.__vectorize, self.data)
        self.modesInfo['user_oriented'] = usersInterestsMatrix
            # usersSimilarities must be computed when each new user comes in.
    
    def __item_oriented_train(self):
        usersInterestsMatrix = map(self.__vectorize, self.data)
        interestsUsersMatrix = [[userInterestsVec[j] for userInterestsVec in usersInterestsMatrix]
                                for j,_ in enumerate(self.uniqueInterests)]
        self.modesInfo['interest_oriented'] = interestsUsersMatrix
        self.interestsSimilarities = self.__mat_cosine(np.array(interestsUsersMatrix))
            # interestsSimilarities can be computed in prior, 
            #  assuming we always have the same list of interests.
            #  for new interests, we have to collect data over all
            #  the users and update current data.
    
    # SIMILARITY COMPUTATION
    
    def __vec_cosine(self, user_i, user_j):
        return np.dot(user_i,user_j) / (np.sqrt(np.dot(user_i,user_i))*np.sqrt(np.dot(user_j,user_j)))
    
    def __mat_cosine(self, matrix):
        matrix_norm = matrix / np.apply_along_axis(lambda r: np.sqrt(np.dot(r,r)), 1, matrix)[:,np.newaxis]
        return np.dot(matrix_norm, matrix_norm.T)
    
    # VECTORIZER
    
    def __vectorize(self, userInterests):
        return [1 if interest in userInterests else 0 for interest in self.uniqueInterests]
    
    # RECOMMENDER
    
    def recommend(self, userInterests=[], mode='most_popular', k=5):
        assert mode in ['most_popular','user_oriented','interest_oriented']
        # TODO: assert  # ensure that mode has been trained.
        if mode=='most_popular':
            for i in xrange(k):
                print "The Rank %d Most Recommended: %s (Popularity Count: %d)" % \
                      (i+1,self.modesInfo[mode][i][0],self.modesInfo[mode][i][1])   
        elif mode=='user_oriented': 
            userInterestsVec = self.__vectorize(userInterests)
            usersSimilarities = [(i,self.__vec_cosine(userInterestsVec,otherUserInterestsVec)) 
                                  for i,otherUserInterestsVec in enumerate(self.modesInfo[mode])] # i: otherUserID
            suggestions = defaultdict(float)
            for i,similarity in usersSimilarities: # i: userID.
                for interest in self.data[i]:
                    suggestions[interest] += similarity
            suggestions = sorted(suggestions.items(),key=lambda(_,weight):weight,reverse=True)
            suggestions = [(suggestion,weight) for suggestion,weight in suggestions
                           if suggestion not in userInterests]
            for i in xrange(k):
                print "The Rank %d Most Recommended: %s (Popularity Weight: %.6f)" % \
                      (i+1,suggestions[i][0],suggestions[i][1])
        else:
            userInterestsVec = self.__vectorize(userInterests)
            suggestions = defaultdict(float) 
            for interest in userInterests:
                if interest in self.uniqueInterests:
                    interestID = self.uniqueInterests.index(interest)
                    for otherInterestID,otherInterest in enumerate(self.uniqueInterests):
                        suggestions[otherInterest] += self.interestsSimilarities[interestID][otherInterestID]
            suggestions = sorted(suggestions.items(),key=lambda(_,weight):weight,reverse=True)
            suggestions = [(suggestion,weight) for suggestion,weight in suggestions
                           if suggestion not in userInterests]
            for i in xrange(k):
                print "The Rank %d Most Recommended: %s (Popularity Weight: %.6f)" % \
                      (i+1,suggestions[i][0],suggestions[i][1])           
        if self.retrain: 
            self.__train()
            retrain = False
