from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import expm, solve
from numpy.linalg import norm

def expEV(n_steps, to, te, y0, f, Df):
    """ Exponentielles Rosenbrock-Euler-Verfahren: O(h^2)

    Input:
        n_steps -- Anzahl Zeitschritte
        y0      -- Startwert zur Zeit t=to
        to      -- Anfangszeit
        te      -- Endzeit
        f       -- Rechte Seite der DGL, dy/dt = f(y).
        Df      -- Jacobi-Matrix

    Output:
        ts     -- [to ... te]
        y      -- Loesung y
    """
    t, dt = linspace(to, te, n_steps, retstep=True)

    # Speicherallokation
    y0 = atleast_1d(y0)
    y = zeros((n_steps, y0.shape[0]))

    # TODO implementieren Sie das exponentielle Euler Verfahren.

    return t, y


if __name__ == '__main__':

    # TODO Jacobi-Matrix
    Df = lambda y: None

    # TODO Rechte Seite
    f = lambda y: None

    # TODO Exakte Loesung
    sol = lambda t: np.zeros((t.size, 2))

    # Anfangswert
    y0 = array([-1, 1])

    to = 0
    te = 6
    nsteps = 20
    ts, y = expEV(nsteps, to, te, y0, f, Df)

    t_ex = linspace(to, te, 1000)
    y_ex = sol(t_ex)

    figure()
    subplot(1,2,1)
    plot(ts, y[:,0], 'r-x', label=r'$y[0]$')
    plot(ts, y[:,1], 'g-x', label=r'$y[1]$')
    plot(t_ex, y_ex[:,0],'r', label=r'$y_{ex}[0$]')
    plot(t_ex, y_ex[:,1],'g', label=r'$y_{ex}[1$]')
    legend(loc='best')
    xlabel('$t$')
    ylabel('$y$')
    grid(True)

    subplot(1,2,2)
    semilogy( ts, norm(y-sol(ts), axis=1), label=r'$|| y - y_{ex}||$')
    xlabel('$t$')
    ylabel('Abs. Fehler')
    legend(loc='best')
    grid(True)
    tight_layout()
    savefig('exp_euler.pdf')
    show()

    # Konvergenzordung

    figure()
    N = [24, 48, 96, 192, 384]
    hs = []  # Gitterweite.
    errors = []  # Fehler.

    # TODO Berechnen Sie die Konvergenzordung.


    # NOTE, die folgenden Zeilen könnten Ihnen beim plotten helfen.
    # loglog(hs, errors)
    # title('Konvergenzplot')
    # gca().invert_xaxis()
    # grid(True)
    # xlabel('$h$')
    # ylabel('Abs. Fehler')
    # savefig('exp_euler_konvergenz.pdf')
    # show()

    # Berechnung der Konvergenzraten
    conv_rate = polyfit(log(hs), log(errors), 1)[0]
    print('Exponentielles Eulerverfahren konvergiert mit algebraischer Konvergenzordnung: %.2f' % conv_rate)
