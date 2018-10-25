# https://bokeh.pydata.org/en/latest/docs/user_guide/plotting.html
import pandas as pd
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.plotting import output_file
from bokeh.plotting import save
from bokeh.sampledata.stocks import AAPL

df = pd.DataFrame(AAPL)
df['date'] = pd.to_datetime(df['date'])

output_file('../output/datetime.html')

# create a new plot with a datetime axis type
# p_open = figure(plot_width=800, plot_height=250, x_axis_type='datetime')
# p_open.line(df['date'], df['open'], color='navy', alpha=0.5)
#
# p_high = figure(plot_width=800, plot_height=250, x_axis_type='datetime')
# p_high.line(df['date'], df['high'], color='green', alpha=0.5)
#
# p_low = figure(plot_width=800, plot_height=250, x_axis_type='datetime')
# p_low.line(df['date'], df['low'], color='red', alpha=0.5)

p_close = figure(plot_width=800, plot_height=250, x_axis_type='datetime')
p_close.line(df['date'], df['close'], color='red', alpha=0.5)

p_volume = figure(plot_width=800, plot_height=250, x_axis_type='datetime')
p_volume.line(df['date'], df['volume'], color='green', alpha=0.5)

# show the results
save(column(p_close, p_volume))
# show(column(p_open, p_high, p_low, p_close, p_volume))
