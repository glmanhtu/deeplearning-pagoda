import sys
from network.download_google_drive import DownloadGoogleDrive


if __name__ == "__main__":
    print 'Argument List:', str(sys.argv)
    file_id = sys.argv[1]
    file_name = sys.argv[2]
    file_location = sys.argv[3]
    google_file = GoogleFile(file_id, file_name, file_location)
    google = DownloadGoogleDrive()
    google.download_file_from_google_drive(google_file)