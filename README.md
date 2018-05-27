# AR_Remote
## Augmented reality, customisable remote control UI in python using aruco markers.
### [DEMO](https://www.youtube.com/watch?v=jeoq_9IZCds&feature=youtu.be)
#### Desc in polish
Własnoręczny interfejs wyświetlany na przekazie na żywo z kamerki internetowej w stylu rozszerzonej rzeczywistości. Korzystam ze biblioteki OpenCV – zbioru algorytmów do przetwarzania obrazu, a w szczególności z aruco – otwartego standardu znaczników dwuwymiarowych (podobnych do kodów QR), zaprojektowanych specjalnie z myślą o wykrywaniu ich na klatce wideo.

Projekt działa w następujący sposób:
1.	Kalibracja kamerki. Wykorzystałem do tego celu zmodyfikowany skrypt z forum opencv, który przez kilka minut zbiera zdjęcia specjalnej szachownicy markerów aruco, żeby zebrać informacje o odkształceniu obrazu widzianego przez moją kamerkę internetową. Te dane posłużą do lepszej estmacji położenia 3d. Przy zmianie urządzenia można albo ponownie wykonać kalibrację, albo użyć domyślnych parametrów.
2.	Wczytanie akcji z pliku .yaml i utworzenie obiektów interfejsu.
3.	Dla każdej klatki z obrazu kamery:
a.	Wykrycie markerów na obrazie za pomocą funkcji z biblioteki cv2.aruco (pod spodem wykonywany jest threshold, canny edge detection i przypasowywanie potencjalnych markerów do słownika markerów danej kategorii (tu 4X41000))

b.	Dla każdego z markerów śledzona jest historia markera o danym id, żeby zapewnić ciągłość między klatkami obrazu pomimo klikuklatkowej utraty (limit 0.1s) Pozwala to na reset menu poprzez chwilowe ukrycie markera, ale problemy z detekcją w danej klatce nie powodują migania markera.

c.	Dane z kalibracji służą do odczytu rotacji i translacji markera w przestrzeni 3d, a następnie zamienione na pitch roll yaw określają, czy wykonywany jest ruch po interfejsie poprzez wychylenie markera.

d.	Na osobnym obrazie rysowany jest aktualny stan interfejsu z uwzględnieniem wykonanych ruchów po interfejsie. 

e.	Ten obraz jest przypasowywany do aktualnie wyświetlanej klatki poprzez znalezienie transformacji perspektywicznej pomiędzy wierzchołkami obrazu interfejsu (kwadrat) a wierzchołkami kwadratu markera (kwadrat w jakimś przekształceniu perspektywicznym i transformacji). Tą transformację stosujemy do obrazu interfejsu, skalujemy go (docelowo ma wystawać poza marker) i dodajemy te obrazy razem (żeby uzyskać efekt hologramu)

f.	Akcje wybrane w interfejsie wysyłamy za pomocą socketów.
