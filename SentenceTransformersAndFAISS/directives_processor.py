import json
import os
import pprint

topics_filename = 'topics_data.json'

def find_next_available_number(ids):
    # Convert string array elements to integers
    int_arr = [int(num) for num in ids]
    
    # Sort the integer array
    int_arr.sort()

    for i in range(len(int_arr)):
        if int_arr[i] != i + 1:
            return i + 1

    return len(int_arr) + 1

class DirectivesProcessor:
    def __init__(self):
        self.topics_data = self.load_topics_data()
        
    # Functions for loading and saving data
    def load_topics_data(self):
        if os.path.exists(topics_filename):
            with open(topics_filename, "r") as file:
                topics_data = json.load(file)
        else:
            topics_data = {"topics": {}}
            save_topics_data(topics_data)

        return topics_data

    def save_topics_data(self):
        with open(topics_filename, "w") as file:
            json.dump(self.topics_data, file, indent=4)            

    # This method looks for both the id and name so that it can be used to 
    # find the topic's info and to also check to see if it already exists.
    def get_topic(self, topic):
        # Search for the ID first
        _topic = self.topics_data["topics"].get(topic)

        if _topic:
            return _topic

        for _, _topic in self.topics_data["topics"].items():
#             print('_topic', _topic)
#             print('_topic.get("topic_title")', _topic.get("topic_title"))
            topic_name = _topic.get("topic_title", "").lower()
            if topic_name == topic.lower():
                return _topic

        return None

    # In order to simplify the process, this method is only attempting to find an existing directive.
    # The original directive ID is sufficient to modify the value
    def get_directive_by_id(self, directive_id, include_topic=False):
    #     global topics_data
        # First, get the topic from the directive ID
        split_id = directive_id.split('.')
        topic_id = split_id[0]
        topic = self.get_topic(topic_id)
        # The topic needs to already exist so that there is an id and name that has already been set
        if topic:
            # Now search for the directive ID
            _directive = topic.get('directives', {}).get(split_id[1])
            if include_topic:
                return topic, _directive
            return _directive

        if include_topic:
            return None, None
        return None

    # This returns the directive_id and its directive
    def get_directive_by_name(self, topic_id, directive_name):
        topic = self.get_topic(topic_id)
        if topic:
            for directive_id, directive in topic.get('directives', {}).items():
                directive_title = directive.get('directive_title', '')
                if directive_title.lower() == directive_name.lower():
                    return directive_id, directive

        return None, None

    # In order to simplify the process, this method is only attempting to find an existing subdirective.
    def get_subdirective(self, directive_id, name_or_id):
    #     global topics_data
        # First, get the directive from the subdirective ID.
        directive = self.get_directive_by_id(directive_id)
        if not directive:
            directive = self.get_directive_by_name(directive_id)

        if directive:
            subdirective = directive.get('subdirectives', {}).get(name_or_id)

            if subdirective:
                # In this case the name_or_id variable is the ID
                return name_or_id, subdirective

            # If you didn't find it by ID, search by name
            for subdirective_id, subdirective in directive.get('subdirectives', {}).items():
                # subdirective should just be a string so we can do a direct comparison at this point
                if name_or_id.lower() == subdirective.lower():
                    return subdirective_id, subdirective

        return None, None

    def add_topic(self, topic_id, topic):
        # Check to see if it already exists
        _topic = self.get_topic(topic_id)
        if not topic_id or not topic:
            raise Exception('A topic requires a topic ID and description')

        if _topic:
            # Don't process a second time if the ID exists
            print("Topic ID " + topic_id + " already exists.")
            return
        _topic = self.get_topic(topic)
        if _topic:
            # Same goes for the name
            print("Topic name " + topic + " already exists.")
            return

        # It's OK to create it
        self.topics_data['topics'][topic_id] = {'topic_title': topic, 'directives': {}}
        self.save_topics_data()

    def add_directive(self, topic_id, directive_title, directive_id=None):
        # Load the topic first
        _topic = self.get_topic(topic_id)
        if not _topic:
            print("Topic ID " + topic_id + " not found.")
            return

        # Make sure that the directive doesn't already exist
        _directive_id, _directive = self.get_directive_by_name(topic_id, directive_title)
    #     print('DIRECTIVE:')
    #     pprint.pprint(_directive)
        if _directive:
            # Don't process a second time if the directive exists
            print("Directive title " + directive_title + " already exists.")
            return _directive_id

        # If you are manually setting the directive_id, make sure that doesn't exist either
        if directive_id:
            _directive = self.get_directive_by_id(topic_id + '.' + directive_id)
            if _directive:
                # Don't process a second time if the directive ID exists
                print("Directive ID " + directive_id + " already exists.")
                return directive_id
        else:
            directive_id = find_next_available_number(_topic['directives'].keys())

        # Hasn't been found so you can create it and save
        # Add to the current topic
    #     print('Updating topic:')
    #     pprint.pprint(_topic, compact=True)
        _topic['directives'][str(directive_id)] = {'directive_title': directive_title, 'subdirectives': {}}

    #     print('New topic:')
    #     pprint.pprint(_topic, compact=True)
        # Update the topic and save
        self.topics_data['topics'][topic_id] = _topic
    #     print('New TOPICS DATA:')
    #     pprint.pprint(topics_data)
        self.save_topics_data()
        # For convenience, return the new ID
        return directive_id

    def add_subdirective(self, directive_id, subdirective_title, subdirective_id=None):
        # directive_id must be of the form <topic_id.directive_id>
        _subdirective_id, _subdirective_title = self.get_subdirective(directive_id, subdirective_title)
        if _subdirective_id:
            print("Subdirective ID " + _subdirective_id + "(" + _subdirective_title + ") already exists.") 
            return _subdirective_id

        # You will need the topic and directive in order to add a subdirective
        _topic, _directive = self.get_directive_by_id(directive_id, True)


        # If you are manually setting the directive_id, make sure that doesn't exist either
        if subdirective_id:
            _subdirective_id, _subdirective = self.get_subdirective(directive_id, subdirective_id)
            if _subdirective:
                # Don't process a second time if the directive ID exists
                print("Subdirective ID " + _subdirective_id + " already exists.")
                return subdirective_id
        else:
    #         directive_id = find_next_available_number(_topic['directives'].keys())
            # Now that you have the directive, get the next available number in the list of subdirectives
            _subdirective_id = find_next_available_number(_directive['subdirectives'].keys())

        # Add the new subdirective to the directive
        print('Saving subdirective with ID', _subdirective_id)
        _directive['subdirectives'][str(_subdirective_id)] = subdirective_title
        # Update the topic
        id_split = directive_id.split('.')
        topic_id = id_split[0]
        _directive_id = id_split[1]
        _topic['directives'][str(_directive_id)] = _directive
        # Update the topic and save
        self.topics_data['topics'][topic_id] = _topic
        self.save_topics_data()  
        # For convenience, return the new ID
        return _subdirective_id

    def get_directive_text_from_directive(self, directive_data):
    #     print('The directive data is: ', directive_data)
        directive_title = directive_data.get('directive_title')
        subdirectives = directive_data.get('subdirectives')
    #     print('The subdirective data is: ', subdirectives)
        text = f"\t{directive_title}\n"
        for key, value in subdirectives.items():
            text += f"\t\t{value}\n"

        return text    

    def get_directive_text(self, topic_id, directive_id):
        directive_data = self.get_directive_by_id(topic_id + '.' + directive_id)
        return self.get_directive_text_from_directive(directive_data)

    def get_topic_text(self, topic_id):
        _topic = self.get_topic(topic_id)
        topic_title = _topic.get('topic_title')
        directives = _topic.get('directives', {})
        text = f"{topic_title}\n"

        for key, directive_data in directives.items():
    #     for directive_data in directives:
            text += self.get_directive_text_from_directive(directive_data)

        return text      

    def get_topic_names(self):
        names = []
        for _name, _topic in self.topics_data["topics"].items():
            names.append(_name)
        return names

    def get_topic_titles(self):
        titles = []
        for _, _topic in self.topics_data["topics"].items():
            titles.append(_topic.get("topic_title"))
        return titles
    
    
    
    