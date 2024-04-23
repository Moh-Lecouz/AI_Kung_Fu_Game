# Creating the architecture of the Convolutional Neural Network

class neural_network(nn.Module):

    def __init__(self, action_size):
          super(neural_network, self).__init__()
          self.conv1 = nn.Conv2d(4, 32, 3, 2)
          self.conv2 = nn.Conv2d(32, 32, 3, 2)
          self.conv3 = nn.Conv2d(32, 32, 3, 2)
          self.flat = nn.Flatten()
          self.fc1 = nn.Linear(512, 128)
          self.fc2a = nn.Linear(128, action_size)
          self.fc2v = nn.Linear(128, 1)

    def forward(self, state):
          x = F.relu(self.conv1(state))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = self.flat(x)
          x = F.relu(self.fc1(x))
          action_values = self.fc2a(x)
          state_value = self.fc2v(x)[0]
          return action_values, state_value