import streamlit as st
import datetime as dt
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

st.title('DQN Stock Trading Agent')

symbol = st.text_input("Enter a stock symbol (e.g., AAPL,TSLA)", "AAPL")
start_date = st.date_input("Start date: yyyy-m-dd", dt.date(2020,2,22))
end_date = st.date_input("Start date: yyyy-m-dd",dt.date(2025,2,22))
initial_balance = st.number_input('Initial Balance ($):', min_value=1000, max_value=1000000, value=10000)

if (end_date - start_date).days < 3 * 365:
    st.warning('Please select a date range of at least 3 years!')

if st.button('Load Data',key= "Load Data"):
    data = yf.download(symbol, start=start_date, end=end_date)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['returns'] = data['Close'].pct_change()

    data.dropna(inplace=True)
    data.reset_index(inplace=True)  
    st.session_state['data'] = data 
    st.write(data) 
    
Actions={ 0:"HOLD", 1:"BUY", 2:"SELL" }

def get_state(data, index):

    return np.array([
        float(data.loc[index, "Close"]),
        float(data.loc[index, "SMA_5"]),
        float(data.loc[index, "SMA_20"]),
        float(data.loc[index, "returns"])
    ])

class TradeEnv:
    def __init__(self,data):
        self.data=data
        self.initial_balance=10000
        self.balance=self.initial_balance
        self.holdings=0
        self.index=0

    def reset(self):
        self.balance=self.initial_balance
        self.holdings=0
        self.index=0
        return get_state(self.data,self.index)
    def step(self,action):
        price=float(self.data.loc[self.index,"Close"])
        reward=0

        if action == 1 and self.balance >= price:
            self.holdings=self.balance//price
            self.balance -= self.holdings * price
        elif action == 2 and self.balance > 0:
            self.balance += self.holdings * price
            self.holdings = 0

        self.index+=1
        done= self.index >= len(self.data)-1

        if done:
            reward= self.balance - self.initial_balance

        next_state=get_state(self.data,self.index) if not done else None

        return next_state,reward,done,{}
    
class DQN(nn.Module):
    def __init__ (self, state_size, action_size):
        super(DQN,self).__init__()
        self.fc1=nn.Linear(state_size, 64)
        self.fc2=nn.Linear(64,64)
        self.fc3=nn.Linear(64,action_size)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size=state_size
        self.action_size=action_size
        self.memory= deque(maxlen=2000)
        self.gamma=0.95
        self.epsilon=1
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.001
        self.model=DQN(state_size,action_size)
        self.optimizer=optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.criterion=nn.MSELoss()

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):

        if random.uniform(0,1) < self.epsilon:
            return random.choice(list(Actions.keys()))

        state=torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self,batch_size):

        if len(self.memory) < batch_size:
            return
        minibatch= random.sample(self.memory,batch_size)

        for state,action,reward,next_state,done in minibatch:
            target = reward
            if not done:
                next_state_tensor=torch.FloatTensor(next_state).unsqueeze(0)
                target += self.gamma * torch.max(self.model(next_state_tensor)).item()

            state_tensor= torch.FloatTensor(state).unsqueeze(0)
            target_tensor= self.model(state_tensor).clone().detach()
            target_tensor[0][action] = target

            self.optimizer.zero_grad()
            output = self.model(state_tensor)
            loss=self.criterion(output, target_tensor)
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_decay


    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()



if 'data' in st.session_state and not st.session_state['data'].empty:
    if st.button('Train Model', key="train_model_button"):
        env = TradeEnv(st.session_state['data'])
        agent = DQNAgent(state_size=4, action_size=3)
        batch_size = 32
        episodes = 300
        total_rewards = []

        for episode in range(episodes):
            progress_text.text(f"Training Progress: Episode {episode+1} / {episodes}")
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            agent.replay(batch_size)
            total_rewards.append(total_reward)

        st.write(f"Training Complete! Final Balance: ${env.balance:.2f}")
        st.write(f"Total Profit: ${env.balance - initial_balance:.2f}")

        if total_rewards:
            plt.figure(figsize=(10, 6))
            plt.plot(total_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Performance')
            st.pyplot(plt)
        else:
            st.warning("No rewards recorded — check if training loop executed properly!")
