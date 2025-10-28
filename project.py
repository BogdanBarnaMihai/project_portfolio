#add the libraries
from sklearn.linear_model import LinearRegression
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from math import sqrt

#part 1 -> welcome phase
print('🙋‍♂️ Welcome. Would you like to:\n')
print('a. Create a New Portfolio?\nb. Load an existing portfolio\n\n(Please type either a,b)')

#ask user what to do
while True:
    while True:
        user_answer = input().lower()
        if user_answer not in ['a','b']:
            print('wrong wording')
        else:
            break
    print('\n')
    n = 0
    #build empty dataframe
    dataframe = pd.DataFrame(columns=['ticker','amount'])

    #if user creates new portfolio
    if user_answer == 'a':
        while True:
            #ask ticker
            print('Enter the stock name in its shortened form (example Apple->AAPL)')
            while True:
                ticker = input().upper()
                data = yf.Ticker(ticker)
                hist = data.history(period='1d')
                if hist.empty:
                    print('wrong format')
                else:
                    break
            #ask amount
            print('\nEnter the amount of stock you have in % (100%=1 and 1%=0.01)')
            amount = float(input())
            new_row = pd.DataFrame([{'ticker': ticker, 'amount': amount}])
            dataframe = pd.concat([dataframe, new_row], ignore_index=True)
            print(f'\nSuccessfully added {ticker} at {amount*100:.2f}%')

            #add more?
            print('\nWould you like to add more? Y/N')
            answer = input().upper()
            if answer == 'N':
                break

        #ask what to analyze
        print('\nWhat would you like to analyze?')
        print('\na.Portfolio summary\nb.Portfolio performance visualization\nc.Portfolio risk metrics\nd.Save or export portfolio report')
        while True:
            user_answer2 = input().lower()
            if user_answer2 not in ['a','b','c','d']:
                print('\nwrong wording')
            else:
                while True:
                    #summary
                    if user_answer2 == 'a':
                        print(dataframe)
                    #visualize
                    elif user_answer2 == 'b':
                        print('\nHow would you like to visualize it?')
                        print('\na.Portfolio cumulative growth over time\nb.Rolling volatility (12-month)\nc.Compare with benchmark (S&P 500)')
                        user_answer3 = input().lower()

                        #predict future prices
                        if user_answer3 == 'a':
                            print('\nHow many months in the future?')
                            months = int(input())
                            for i in range(len(dataframe)):
                                tick = dataframe.loc[i, 'ticker']
                                data = yf.download(tick, start='2025-01-01', end='2025-10-25')
                                price = data['Close'].values
                                X = np.arange(len(price)).reshape(-1, 1)
                                y = price
                                model = LinearRegression()
                                model.fit(X, y)
                                future_dates = np.arange(len(price), len(price) + months*30).reshape(-1, 1)
                                p_p = model.predict(future_dates)
                                predicted = p_p[-1].item()
                                print(f'{tick}: predicted price in {months} months is {predicted:.2f}')
                                plt.plot(X, y)
                                plt.plot(future_dates, p_p, linestyle='dashed', linewidth=2)
                            plt.show()

                        #volatility
                        elif user_answer3 == 'b':
                            for i in range(len(dataframe)):
                                tick = dataframe.iloc[i]['ticker']
                                data = yf.download(tick, start='2025-01-01', end='2025-10-24', auto_adjust=False)['Close']
                                m_p = data.resample('ME').last().ffill()
                                m_c = m_p.pct_change().dropna()
                                m_v = m_c.std()
                                print(f'the volitility for {tick} is {m_v:.4f}')

                        #compare with S&P 500
                        elif user_answer3 == 'c':
                            for i in range(len(dataframe)):
                                ticker = dataframe.loc[i, 'ticker']
                                while True:
                                    ok = input('\nWould you like to visualize it in the future? Y/N').upper()
                                    data = yf.download(ticker, start='2025-01-01', end='2025-10-27')
                                    if ok == 'Y':
                                        months_in_future = int(input('How many months in the future?'))
                                        dat = data['Close'].values
                                        X = np.arange(len(dat)).reshape(-1, 1)
                                        y = dat
                                        model = LinearRegression()
                                        model.fit(X, y)
                                        future_dates = np.arange(len(dat), len(dat)+months_in_future*30).reshape(-1,1)
                                        prediction = model.predict(future_dates)
                                        plt.figure(figsize=(5,3))
                                        plt.plot(X,y)
                                        plt.plot(future_dates,prediction)
                                        plt.title(ticker)
                                        plt.show()
                                        print(f'prediction is {prediction[-1]:.2f}')
                                        break
                                    elif ok == 'N':
                                        plt.figure(figsize=(5,3))
                                        plt.plot(data['Close'].values)
                                        plt.title(ticker)
                                        plt.show()
                                        break
                                    else:
                                        print('wrong format')
                            #plot S&P 500
                            data2 = yf.download('^GSPC', start='2025-01-01', end='2025-10-27')
                            plt.figure(figsize=(5,3))
                            plt.plot(data2['Close'].values)
                            plt.title('S&P 500')
                            plt.show()

                    #risk metrics
                    elif user_answer2 == 'c':
                        s1 = 0
                        s2 = 1
                        cov = 1
                        standard = 1
                        n = 0
                        for i in range(len(dataframe)):
                            amount = dataframe.loc[i, 'amount']
                            tick = dataframe.loc[i, 'ticker']
                            data = yf.download(tick, start='2025-01-01', end='2025-10-24', auto_adjust=False)['Close']
                            m_p = data.resample('ME').last().ffill()
                            m_c = m_p.pct_change().dropna()
                            s_v = (m_c.iloc[0]-m_c.iloc[1])/m_c.iloc[1]
                            m_v = m_c.std()
                            Rx = m_c.iloc[:10].sum()
                            s1 += (amount*amount)*(m_v*m_v)
                            Cov = s_v - Rx
                            standard *= m_v
                            s2 *= amount*m_v
                            cov *= Cov
                            n += 1
                        True_cov = cov/(n-1) if n>1 else cov/n
                        True_standard = True_cov/standard
                        True_s2 = 2*s2*True_standard
                        Portfolio_o = sqrt(abs(s1+True_s2))
                        print(f"{Portfolio_o:.2f}")

                    #save/export
                    elif user_answer2 == 'd':
                        cout = input('\nhow would you like to save it?')
                        dataframe.to_csv(f'{cout}.csv', index=False)

                    #do more?
                    print('\nWould you like to do more? Y/N')
                    ok = input().upper()
                    if ok == 'N':
                        os._exit(0)
                    else:
                        print('\nWhat would you like to analyze?')
                        print('\na.Portfolio summary\nb.Portfolio performance visualization\nc.Portfolio risk metrics\nd.Save or export portfolio report\ne.Go back to main menu')
                        break
     #part b -> load existing portfolio
    elif user_answer == 'b':
        #ask for csv path
        name_path = input('Enter the path of the csv')
        dataframe = pd.read_csv(name_path)

        #ask what to analyze
        print('\nWhat would you like to analyze?')
        print('\na.Portfolio summary\nb.Portfolio performance visualization\nc.Portfolio risk metrics\nd.Save or export portfolio report')
        
        while True:
            user_answer2 = input().lower()
            if user_answer2 not in ['a','b','c','d','e']:
                print('\nwrong wording')
            else:
                while True:
                    #summary
                    if user_answer2 == 'a':
                        print(dataframe)
                    
                    #visualization
                    elif user_answer2 == 'b':
                        print('\nHow would you like to visualize it?')
                        print('\na.Portfolio cumulative growth over time\nb.Rolling volatility (12-month)\nc.Compare with benchmark (S&P 500)')
                        user_answer3 = input().lower()

                        #cumulative growth prediction
                        if user_answer3 == 'a':
                            print('\nHow many months in the future?')
                            months = int(input())
                            for i in range(len(dataframe)):
                                tick = dataframe.loc[i, 'ticker']
                                data = yf.download(tick, start='2025-01-01', end='2025-10-25')
                                price = data['Close'].values
                                X = np.arange(len(price)).reshape(-1, 1)
                                y = price
                                model = LinearRegression()
                                model.fit(X, y)
                                future_dates = np.arange(len(price), len(price) + months*30).reshape(-1, 1)
                                p_p = model.predict(future_dates)
                                predicted = p_p[-1].item()
                                print(f'{tick}: predicted price in {months} months is {predicted:.2f}')
                                plt.plot(X, y)
                                plt.plot(future_dates, p_p, linestyle='dashed', linewidth=2)
                            plt.show()

                        #rolling volatility
                        elif user_answer3 == 'b':
                            for i in range(len(dataframe)):
                                tick = dataframe.iloc[i]['ticker']
                                data = yf.download(tick, start='2025-01-01', end='2025-10-24', auto_adjust=False)['Close']
                                m_p = data.resample('ME').last().ffill()
                                m_c = m_p.pct_change().dropna()
                                m_v = m_c.std()
                                print(f'the volitility for {tick} is {m_v:.4f}')

                        #compare with S&P 500
                        elif user_answer3 == 'c':
                            for i in range(len(dataframe)):
                                ticker = dataframe.loc[i, 'ticker']
                                while True:
                                    ok = input('\nWould you like to visualize it in the future? Y/N').upper()
                                    data = yf.download(ticker, start='2025-01-01', end='2025-10-27')
                                    if ok == 'Y':
                                        months_in_future = int(input('How many months in the future?'))
                                        dat = data['Close'].values
                                        X = np.arange(len(dat)).reshape(-1, 1)
                                        y = dat
                                        model = LinearRegression()
                                        model.fit(X, y)
                                        future_dates = np.arange(len(dat), len(dat)+months_in_future*30).reshape(-1,1)
                                        prediction = model.predict(future_dates)
                                        plt.figure(figsize=(5,3))
                                        plt.plot(X, y)
                                        plt.plot(future_dates, prediction)
                                        plt.title(ticker)
                                        plt.show()
                                        print(f'prediction is {prediction[-1]:.2f}')
                                        break
                                    elif ok == 'N':
                                        plt.figure(figsize=(5,3))
                                        plt.plot(data['Close'].values)
                                        plt.title(ticker)
                                        plt.show()
                                        break
                                    else:
                                        print('wrong format')
                            #plot S&P 500
                            data2 = yf.download('^GSPC', start='2025-01-01', end='2025-10-27')
                            plt.figure(figsize=(5,3))
                            plt.plot(data2['Close'].values)
                            plt.title('S&P 500')
                            plt.show()

                    #risk metrics
                    elif user_answer2 == 'c':
                        s1 = 0
                        s2 = 1
                        cov = 1
                        standard = 1
                        n = 0
                        for i in range(len(dataframe)):
                            amount = dataframe.loc[i, 'amount']
                            tick = dataframe.loc[i, 'ticker']
                            data = yf.download(tick, start='2025-01-01', end='2025-10-24', auto_adjust=False)['Close']
                            m_p = data.resample('ME').last().ffill()
                            m_c = m_p.pct_change().dropna()
                            s_v = (m_c.iloc[0]-m_c.iloc[1])/m_c.iloc[1]
                            m_v = m_c.std()
                            Rx = m_c.iloc[:10].sum()
                            s1 += (amount*amount)*(m_v*m_v)
                            Cov = s_v - Rx
                            standard *= m_v
                            s2 *= amount*m_v
                            cov *= Cov
                            n += 1
                        True_cov = cov/(n-1) if n>1 else cov/n
                        True_standard = True_cov/standard
                        True_s2 = 2*s2*True_standard
                        Portfolio_o = sqrt(abs(s1+True_s2))
                        print(f"{Portfolio_o:.2f}")

                    #save/export
                    elif user_answer2 == 'd':
                        cout = input('\nhow would you like to save it?')
                        dataframe.to_csv(f'{cout}.csv', index=False)

                    #do more?
                    print('\nWould you like to do more? Y/N')
                    ok = input().upper()
                    if ok == 'N':
                        os._exit(0)
                    else:
                        print('\nWhat would you like to analyze?')
                        print('\na.Portfolio summary\nb.Portfolio performance visualization\nc.Portfolio risk metrics\nd.Save or export portfolio report\ne.Go back to main menu')
                        break
